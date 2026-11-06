"""Deterministic CPU checks for the repaired contracts; no model or GPU jobs."""
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from contracts import TASK_POINTS, canonical_to_world, coordinate_contract, task_mask, world_to_canonical
from guidance import attach, iteration_counts
from runtime import load_config, sha


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


class Contracts(unittest.TestCase):
    def test_training_coordinate_convention_and_inverse(self):
        from importlib.util import module_from_spec, spec_from_file_location

        points = np.tile([0., 0., .45], (22, 1))
        points[0, 2] = .84
        points[[1, 16], 1] = .12
        points[[2, 17], 1] = -.12
        points[[10, 11], 2] = .09707717480258388
        theta = .67
        c, s = np.cos(theta), np.sin(theta)
        points = points @ np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]]) + [2., -.7, .01]
        transform = coordinate_contract(points, [f"joint_{i}" for i in range(29)], "u" * 64)
        encoded = world_to_canonical(points, transform)
        encoder_source = HERE.parent / "feature263.py"
        if encoder_source.is_file():
            spec = spec_from_file_location("_training_encoder", encoder_source)
            encoder = module_from_spec(spec)
            spec.loader.exec_module(encoder)
            expected = encoder.canonicalize_z_up(np.stack([points, points]))[0]
            np.testing.assert_allclose(encoded, expected, atol=1e-12)
        np.testing.assert_allclose(canonical_to_world(encoded.astype(np.float32), transform), points, atol=1e-6)
        np.testing.assert_allclose(np.array(transform["canonical_to_world_rotation"])[:, 1], [0, 0, 1])
        self.assertAlmostEqual(transform["coordinate_provenance"]["height_origin_world_m"], .1070771748)
        self.assertAlmostEqual(encoded[0, 1], .742922825197416)
        np.testing.assert_allclose(encoded[[10, 11], 1], 0.)

    def test_train_and_sample_share_exact_task_layouts(self):
        expected_points = {"pick": [0, 4, 5, 19, 21], "pour": [0, 10, 11, 21],
                           "bimanual_box_lift": [0, 10, 11, 20, 21],
                           "box_push_pull": [0, 10, 11, 20, 21]}
        for task, (dense, sparse) in TASK_POINTS.items():
            for frames in (1, 7, 120, 180, 199):
                mask = task_mask(task, frames)
                self.assertEqual(mask.shape, (frames, 22))
                np.testing.assert_array_equal(np.flatnonzero(mask.any(0)), expected_points[task])
                self.assertTrue(mask[:, dense].all())
                self.assertTrue(mask[::5, sparse].all())
                self.assertEqual(int(mask.sum()), frames * len(dense) + ((frames + 4) // 5) * len(sparse))

    def test_exact_upstream_schedule_boundaries(self):
        self.assertEqual(iteration_counts([0, 9, 10, 19, 20, 999], True), (100, 100, 100, 100, 20, 20))
        self.assertEqual(iteration_counts([0, 9, 10, 19, 20, 999], False), (500, 500, 10, 10, 10, 10))

    def test_python310_syntax_and_isolated_run_config(self):
        for name in ("contracts.py", "conditions.py", "runtime.py", "guidance.py", "validation.py", "train.py", "sample.py"):
            ast.parse((HERE / name).read_text(encoding="utf8"), filename=name, feature_version=(3, 10))
        root = PROJECT.resolve()
        baseline = HERE / "runtime.py"
        new = HERE / "contracts.py"
        baseline_config = {"project_root": str(root), "stage_dir": str(root / "old"),
                           "run_name": "old", "max_steps": 10000}
        config = {"baseline_config": str(baseline), "stage_dir": str(root / "new"), "run_name": "new"}
        reads = (json.dumps(config), json.dumps(baseline_config))
        with patch.object(Path, "read_text", side_effect=reads):
            self.assertEqual(load_config(new)["stage_dir"], str(root / "new"))
        self.assertEqual(sha(new), hashlib.sha256(new.read_bytes()).hexdigest())
        for change in ({"stage_dir": str(root / "old")}, {"stage_dir": str(root / "old" / "nested")},
                       {"run_name": ".."}, {"max_steps": 10001}, {"max_sample_batch_size": 33}):
            reads = (json.dumps({**config, **change}), json.dumps(baseline_config))
            with patch.object(Path, "read_text", side_effect=reads):
                with self.assertRaises(ValueError):
                    load_config(new)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy unavailable in this CPU runtime")
    def test_pick_plan_uses_randomized_contact_frame_and_preserves_goal(self):
        from conditions import build

        initial = np.zeros((22, 3))
        initial[19] = [.04, -.16, .76]
        initial[21] = [.20, -.16, .76]
        rotations = np.repeat(np.eye(3)[None], 22, axis=0)
        scene = {"object_center": [.35, -.2, .90]}
        result = build("pick", range(100), 20261104, initial, rotations, scene, "abc")
        self.assertEqual(set(result["grasp_frames"]), set(range(40, 51)))
        self.assertEqual(result["desired_wrist_quat_wxyz"].shape, (100, 120, 2, 4))
        for index, frame in enumerate(result["grasp_frames"]):
            np.testing.assert_allclose(result["world"][index, frame, 21],
                                       result["object_initial"][index] + [0, 0, .04])
            self.assertEqual(result["planned_hand_closed"][index, frame - 1, 1], 0)
            self.assertEqual(result["planned_hand_closed"][index, frame, 1], 1)
            self.assertAlmostEqual(np.linalg.norm(result["world"][index, frame, 21]
                                                  - result["world"][index, frame, 19]), .16)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable in this CPU runtime")
    def test_guidance_under_no_grad_with_mixed_timesteps_empty_hints_and_padding(self):
        import torch

        def recover(features, joints):
            return features[..., :3].unsqueeze(-2).expand(-1, -1, joints, -1)

        module = "data_loaders.humanml.scripts.motion_process"
        fake = SimpleNamespace(mean=torch.zeros(263), std=torch.ones(263),
                               raw_mean=torch.tensor([.8, 0., 0.] * 22), raw_std=torch.ones(66),
                               posterior_log_variance_clipped=np.log(np.full(1000, .001)))
        with patch.dict(sys.modules, {module: SimpleNamespace(recover_from_ric=recover)}):
            attach(fake)
        x = torch.zeros((3, 263, 1, 3))
        x[:, 0, :, :] = .1
        mask = torch.zeros((3, 3, 22), dtype=torch.bool)
        mask[:2, :, 0] = True
        y = {"hint_mask": mask, "hint": torch.zeros((3, 3, 66)),
             "mask": torch.tensor([True, True, False]).expand(3, 1, 1, 3)}
        with torch.no_grad():
            result = fake.guide(x, torch.tensor([20, 19, 500]), {"y": y}, train=True)
        torch.testing.assert_close(result[:, 0, 0, 1], torch.tensor([.2, .6, .1]))
        torch.testing.assert_close(result[..., 0], x[..., 0])
        torch.testing.assert_close(result[..., 2], x[..., 2])
        torch.testing.assert_close(x[:, 0, 0, 1], torch.full((3,), .1))
        self.assertFalse(result.requires_grad)

        fake.raw_mean[0] = 150
        with torch.no_grad():
            early = fake.guide(x, torch.tensor([10, 9, 500]), {"y": y})
        # Two valid frames constrain one point and set the sampling density to 2,
        # even though the position gradient of frame 0 is held fixed.
        torch.testing.assert_close(early[:, 0, 0, 1], torch.tensor([1.1, 50.1, .1]), atol=.001, rtol=0.)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable in this CPU runtime")
    def test_full_checkpoint_readback_checks_adam_rng_and_tensor_bytes(self):
        import torch
        from validation import buffer_witness, compare_exact, verified_readback

        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        for _ in range(2):
            optimizer.zero_grad()
            model(torch.ones((1, 3))).square().sum().backward()
            optimizer.step()
        payload = {"version": 2, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "cursor": {"step": 2}, "config": {"lr": .001}, "provenance": {"tag": "cpu_fixture"},
                   "rng": {"torch": torch.get_rng_state(), "sequence": (1, [2, 3])}}
        size, witness = buffer_witness(payload, 1024 * 1024)
        self.assertGreater(size, 0)
        self.assertEqual(witness["adam_steps"]["parameter_update_histogram"], {"2": 2})
        self.assertEqual(witness["compared_tensor_counts"]["optimizer"], 6)
        with self.assertRaisesRegex(ValueError, "tensor bytes changed"):
            compare_exact(torch.tensor([0.]), torch.tensor([-0.]))
        with io.BytesIO() as buffer:
            torch.save(payload, buffer)
            buffer.seek(0)
            next(iter(payload["optimizer"]["state"].values()))["exp_avg"].add_(1)
            with self.assertRaisesRegex(ValueError, "exp_avg.*tensor bytes changed"):
                verified_readback(payload, buffer)


if __name__ == "__main__":
    unittest.main(verbosity=2)


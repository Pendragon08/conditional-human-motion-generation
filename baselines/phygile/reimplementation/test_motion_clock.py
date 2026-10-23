"""Clock, frame and proposer counterexamples; oracle output is NOT learned motion."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
import torch
from codec262 import G1Codec, SCHEMA, quat_matrix, matrix_quat, rotvec_matrix
from features import BlockNormalizer
from physics_prefix import Motion
from motion_clock import (CONTRACT, validate_feature_contract, manifest_feature_contract,
                          resample_native, prepare_context, generation_grid)
from diffusion_proposer import DiffusionProposer

ROOT = Path(__file__).resolve().parents[1]


def trajectory(codec, times):
    times = np.asarray(times)
    state = np.zeros((len(times), 71))
    state[:, :3] = np.stack([2+.1*times, -3+.04*times, .8+times*0], -1)
    rotation = rotvec_matrix(np.stack([times*0, times*0, 2.8+.6*times], -1))
    rotation = rotation @ rotvec_matrix([.12, .08, 0])
    state[:, 3:7] = matrix_quat(rotation)
    state[:, 7:10] = [.1, .04, 0]
    state[:, 10:13] = [0, 0, .6]
    state[:, 13:42] = (codec.low+codec.high)/2
    return state


def motion(codec, native, fps=50.):
    positions, _ = codec.fk(native[:, :3], native[:, 3:7], native[:, 13:42])
    bodies = np.stack([positions[n] for n in codec.body_names], 1)
    return Motion(native, bodies, fps, 'root13_q29_dq29_wxyz_world', codec.body_names)


class ClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.codec = G1Codec(ROOT/'configs/g1_codec_skeleton.json')
        torch.set_num_threads(4)

    def test_legacy_missing_invalid_or_wrong_frame_contract_rejected(self):
        good = {'version': CONTRACT, 'feature_fps': 20., 'coordinate_frame': 'initial_root_xy_yaw'}
        self.assertEqual(validate_feature_contract(good), 20.)
        for bad in [None, {}, {**good, 'version': 'unknown'},
                    *[{**good, 'feature_fps': rate} for rate in [0, -1, True, '20', float('nan')]],
                    {**good, 'coordinate_frame': 'world'}]:
            with self.assertRaises(ValueError):
                validate_feature_contract(bad)

    def test_manifest_binds_all_split_sidecars_and_rejects_mixed_clocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, meta = self.codec.encode(trajectory(self.codec, np.arange(4)/20), 20.)
            manifest = {'purpose': 'research_training', 'codec_schema': SCHEMA,
                'codec_skeleton_sha256': meta['skeleton_sha256'], 'splits': {}}
            for split in ['train', 'test']:
                (root/f'{split}.json').write_text(json.dumps(meta))
                manifest['splits'][split] = [{'codec_metadata': f'{split}.json', 'sha256': split}]
            result = manifest_feature_contract(manifest, root/'manifest.json')
            self.assertEqual(result['feature_fps'], 20.)
            self.assertEqual(len(result['metadata']), 2)
            (root/'test.json').write_text(json.dumps({**meta, 'fps': 50.}))
            with self.assertRaisesRegex(ValueError, 'Mixed feature clocks'):
                manifest_feature_contract(manifest, root/'manifest.json')
            (root/'test.json').write_text(json.dumps({**meta, 'initial_root_pos': [1, 0, .8]}))
            with self.assertRaisesRegex(ValueError, 'initial root XY/yaw'):
                manifest_feature_contract(manifest, root/'manifest.json')

    def test_translation_rotation_speed_and_quaternion_sign(self):
        times = np.arange(41)/20
        native = trajectory(self.codec, times)
        native[1::2, 3:7] *= -1
        query = np.arange(101)/50
        rebuilt = resample_native(native, times, query)
        truth = trajectory(self.codec, query)
        np.testing.assert_allclose(rebuilt[:, :3], truth[:, :3], atol=1e-12)
        np.testing.assert_allclose(quat_matrix(rebuilt[:, 3:7]), quat_matrix(truth[:, 3:7]), atol=1e-12)
        np.testing.assert_allclose(rebuilt[:, 7:13], truth[:, 7:13], atol=1e-11)
        with self.assertRaisesRegex(ValueError, 'extrapolation'):
            resample_native(native, times, np.arange(102)/50)
        with self.assertRaisesRegex(ValueError, 'uniform clock'):
            resample_native(native, times, np.array([0, .02, .05]))
        with self.assertRaisesRegex(ValueError, 'Zero root quaternion'):
            resample_native(np.zeros_like(native), times, query)

    def test_context_ends_at_actual_boundary_and_rotates_velocity(self):
        prefix = motion(self.codec, trajectory(self.codec, np.arange(52)/50))
        features, meta, clock = prepare_context(self.codec, prefix, 20., canonicalize=True)
        self.assertEqual(clock['context_sample_times_relative_to_boundary_s'][-1], 0)
        self.assertEqual(len(features), 21)
        transform = np.asarray(meta['canonical_rotation'])
        np.testing.assert_allclose(features[-1, :3], transform@prefix.states[-1, 10:13], atol=1e-7)
        np.testing.assert_allclose(features[-1, 3:6], transform@prefix.states[-1, 7:10], atol=1e-7)
        self.assertGreater(np.max(np.abs(features[-1, 3:6]-prefix.states[-1, 7:10])), .1)
        cold = prefix.slice(-1)
        _, _, cold_clock = prepare_context(self.codec, cold, 20., canonicalize=True)
        self.assertEqual(cold_clock['held_initialization_intervals'], 1)
        self.assertFalse(cold_clock['actual_history_invented'])

    def test_grid_extra_sample_is_generated_not_extrapolated_and_final_rejects(self):
        feature, control = generation_grid(7, 50., 20., False)
        self.assertAlmostEqual(feature[-1], .15)
        self.assertAlmostEqual(control[-1], .14)
        feature, control = generation_grid(50, 50., 20., True)
        self.assertAlmostEqual(feature[-1], control[-1])
        with self.assertRaisesRegex(ValueError, 'off the feature grid'):
            generation_grid(51, 50., 20., True)

    def test_proposer_oracle_exact_prefix_world_pose_and_paper_cfg(self):
        # Only the sampler is replaced by an analytical oracle. Execute the real
        # conditioning/normalization/codec/resampling/projection/proposer path.
        codec = self.codec
        prefix = motion(codec, trajectory(codec, np.arange(51)/50))
        suffix = motion(codec, trajectory(codec, [2.]))

        class Oracle:
            calls = []

            def sample(self, shape, tokens, text_valid, scale, known_values, known_mask, generator):
                self.calls.append({'scale': scale, 'shape': shape})
                clean, _ = codec.encode(trajectory(codec, np.arange(shape[1])/20), 20.)
                clean = torch.from_numpy(clean)[None]
                return torch.where(known_mask, known_values, clean)

        with tempfile.TemporaryDirectory() as tmp:
            proposer = object.__new__(DiffusionProposer)
            proposer.codec = codec
            proposer.feature_fps = 20.
            proposer.canonicalize = True
            proposer.cfg_scale = 2.5
            proposer.device = torch.device('cpu')
            proposer.seed = 17
            proposer.calls = []
            proposer.output = Path(tmp)
            proposer.normalizer = BlockNormalizer(np.zeros(262), np.ones(262))
            proposer.tokens = torch.zeros(1, 1, 768)
            proposer.text_valid = torch.ones(1, 1, dtype=torch.bool)
            proposer.diffusion = Oracle()
            for new_frames, final in [(50, False), (7, False), (50, True)]:
                candidate = proposer(prefix, suffix, new_frames, final, 1)
                self.assertEqual(len(candidate), len(prefix)+new_frames)
                np.testing.assert_array_equal(candidate.states[:len(prefix)], prefix.states)
                np.testing.assert_array_equal(candidate.body_positions[:len(prefix)], prefix.body_positions)
                expected = trajectory(codec, 1+np.arange(1, new_frames+1)/50)
                np.testing.assert_allclose(candidate.states[len(prefix):, :3], expected[:, :3], atol=2e-6)
                np.testing.assert_allclose(quat_matrix(candidate.states[len(prefix):, 3:7]),
                                           quat_matrix(expected[:, 3:7]), atol=2e-6)
                if final:
                    np.testing.assert_array_equal(candidate.states[-1:], suffix.states)
                    np.testing.assert_array_equal(candidate.body_positions[-1:], suffix.body_positions)
                self.assertEqual(proposer.diffusion.calls[-1]['scale'], 2.5)
                self.assertFalse(proposer.calls[-1]['clock']['time_stretched'])
            self.assertEqual(proposer.diffusion.calls[0]['shape'], (1, 41, 262))
            count = len(proposer.calls)
            with self.assertRaisesRegex(ValueError, 'off the feature grid'):
                proposer(prefix, suffix, 51, True, 1)
            self.assertEqual(len(proposer.calls), count)
            with self.assertRaisesRegex(ValueError, 'one-frame terminal suffix'):
                proposer(prefix, suffix.append(suffix), 50, True, 1)


if __name__ == '__main__':
    unittest.main()

"""Synthetic smoke only: real legacy DDPM, tiny fake model and fake solver."""
import json
from types import SimpleNamespace
import unittest
import numpy as np
import test_generate_robot_prior as generation_fixture
from postopt_generated_bundle import postopt_generated_bundle


class SyntheticChainTest(unittest.TestCase):
    def test_generation_to_joint_bundle_preserves_alignment_and_provenance(self):
        fixture = generation_fixture.GenerationTest(); fixture.setUp()
        self.addCleanup(fixture.tearDown)
        generated = fixture.run_generation()
        seen = {}
        def detect(points, thres):
            seen['points'] = points.copy()
            return np.ones((1, points.shape[1] - 1)), np.zeros((1, points.shape[1] - 1))
        def solve(**kw):
            seen['target'] = kw['target_keypoints'].copy()
            n = len(seen['target'])
            return SimpleNamespace(wxyz_xyz=np.tile([1.,0,0,0,0,0,0], (n,1))), np.zeros((n,27))
        class Ground:
            def project_points(self, points):
                points = np.array(points, copy=True); points[:,2] = 0; return points
        names = [f'j{i}' for i in range(27)]
        output = fixture.root / 'joints.npz'
        postopt_generated_bundle(input_path=fixture.out, input_sha256=generated['output_sha256'],
            output_path=output, sample_index=0, solver_provenance={'solver':'fake synthetic test'}, min_free_bytes=0,
            solver=solve, foot_detector=detect, robot=SimpleNamespace(joints=SimpleNamespace(actuated_names=names)),
            robot_collision=None, heightmap=Ground(), human_indices=[0], robot_indices=[0], connection_mask=[True],
            weights={}, canonical_to_solver_rotation=np.eye(3), canonical_to_solver_translation=[0,0,0],
            pick_point_solver=[0,0,0], foot_displacement_sq_threshold=.001, joint_names=names,
            joint_lower=np.full(27,-1.), joint_upper=np.full(27,1.), fps=20.)
        with np.load(fixture.out, allow_pickle=False) as prior, np.load(output, allow_pickle=False) as motion:
            np.testing.assert_array_equal(seen['points'][0], prior['canonical_points'][0])
            np.testing.assert_array_equal(seen['target'], prior['canonical_points'][0,1:])
            np.testing.assert_array_equal(motion['input_frame_indices'], np.arange(1,6))
            self.assertEqual(motion['joint_pos'].shape, (5,27))
            meta = json.loads(motion['metadata_utf8'].tobytes())
            self.assertEqual(meta['input_sha256'], generated['output_sha256'])
            self.assertEqual(meta['input_metadata']['checkpoint']['step'], 1)
            self.assertFalse(meta['training_or_physical_acceptance'])


if __name__ == '__main__': unittest.main()

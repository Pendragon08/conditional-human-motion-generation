import unittest
from types import SimpleNamespace
import numpy as np
from robot_prior_postopt import postopt_canonical_motion


class Ground:
    def project_points(self, points):
        out = np.array(points, copy=True)
        out[:, 2] = 0
        return out


class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.points = np.arange(4 * 22 * 3, dtype=np.float32).reshape(4, 22, 3) / 100
        self.calls = {}
        self.rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        def detector(p, thres):
            self.calls['detector'] = (p.copy(), thres)
            return np.array([[0, 1, 2]]), np.array([[2, 0, 1]])
        def solver(**kw):
            self.calls['solver'] = kw
            return SimpleNamespace(wxyz_xyz=np.tile([1., 0, 0, 0, 0, 0, 0], (3, 1))), np.zeros((3, 27))
        self.kw = dict(solver=solver, foot_detector=detector,
                       robot=SimpleNamespace(joints=SimpleNamespace(actuated_names=[f'j{i}' for i in range(27)])), robot_collision='collision',
                       heightmap=Ground(), human_indices=[1], robot_indices=[2], connection_mask=[True],
                       weights={'floor_contact': 1}, canonical_to_solver_rotation=self.rotation,
                       canonical_to_solver_translation=[2., 3., 4.], pick_point_solver=[0., 0., 0.],
                       foot_displacement_sq_threshold=0.002, joint_names=[f'j{i}' for i in range(27)],
                       joint_lower=np.full(27, -1.), joint_upper=np.full(27, 1.), fps=20.)

    def test_transform_alignment_and_callbacks(self):
        out = postopt_canonical_motion(self.points, **self.kw)
        expected = self.points @ self.rotation.T + [2, 3, 4]
        np.testing.assert_allclose(self.calls['detector'][0][0], expected, atol=1e-6)
        np.testing.assert_allclose(self.calls['solver']['target_keypoints'], expected[1:], atol=1e-6)
        np.testing.assert_array_equal(out['input_frame_indices'], [1, 2, 3])
        np.testing.assert_array_equal(out['left_foot_low_speed_proxy'], [False, True, True])
        np.testing.assert_array_equal(out['right_foot_low_speed_proxy'], [True, False, True])
        np.testing.assert_array_equal(self.calls['solver']['left_foot_keypoints'][:, 2], 0)
        self.assertFalse(out['metadata']['training_or_physical_acceptance'])
        self.assertEqual(out['metadata']['max_joint_limit_violation_rad'], 0)

    def test_rejects_reflections_and_bad_output(self):
        self.kw['canonical_to_solver_rotation'] = np.diag([-1, 1, 1])
        with self.assertRaises(ValueError): postopt_canonical_motion(self.points, **self.kw)
        self.assertEqual(self.calls, {})
        self.kw['canonical_to_solver_rotation'] = self.rotation
        self.kw['joint_names'] = list(reversed(self.kw['joint_names']))
        with self.assertRaises(ValueError): postopt_canonical_motion(self.points, **self.kw)
        self.kw['joint_names'] = list(reversed(self.kw['joint_names']))
        self.kw['solver'] = lambda **kw: (SimpleNamespace(wxyz_xyz=np.zeros((3, 7))), np.zeros((3, 27)))
        with self.assertRaises(ValueError): postopt_canonical_motion(self.points, **self.kw)

    def test_limit_violation_reported_without_clipping(self):
        self.kw['solver'] = lambda **kw: (SimpleNamespace(wxyz_xyz=np.tile([1., 0, 0, 0, 0, 0, 0], (3, 1))), np.full((3, 27), 1.2))
        out = postopt_canonical_motion(self.points, **self.kw)
        self.assertAlmostEqual(out['metadata']['max_joint_limit_violation_rad'], 0.2)
        self.assertEqual(out['joint_pos'][0, 0], 1.2)


if __name__ == '__main__': unittest.main()

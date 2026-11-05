import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from g1_fk import URDFFK
from motion_bridge import prepare_fitted_clip, prepare_near_locked_clip, prepare_root_fitted_clip


class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.model = URDFFK(Path(__file__).parents[1] / "assets" / "g1_27dof.urdf")
        self.motion = np.zeros((9, 36))
        self.motion[:, 6] = 1

    def test_decimation_and_fk(self):
        result = prepare_near_locked_clip(self.motion, self.model, source_fps=60,
                                          target_fps=20, max_locked_angle_rad=0.0)
        self.assertEqual(result["joint_pos"].shape, (3, 27))
        self.assertEqual(result["points_z_up_world"].shape, (3, 22, 3))
        self.assertTrue(np.isfinite(result["points_z_up_world"]).all())

    def test_nonlocked_waist_is_rejected_before_downsampling(self):
        self.motion[1, 20] = 0.05
        with self.assertRaisesRegex(ValueError, "re-optimize"):
            prepare_near_locked_clip(self.motion, self.model, source_fps=60,
                                     target_fps=20, max_locked_angle_rad=0.01)

    def test_fitted_clip_requires_explicit_joint_mapping(self):
        motion = self.motion[::3]
        q = np.zeros((len(motion), 27))
        out = prepare_fitted_clip(motion, q, tuple(self.model.actuated), self.model)
        self.assertEqual(out['points_z_up_world'].shape, (3, 22, 3))
        with self.assertRaisesRegex(ValueError, 'joint names'):
            prepare_fitted_clip(motion, q, ('bad',) * 27, self.model)

    def test_root_fit_uses_corrected_root_and_preserves_source(self):
        motion = self.motion[::3].copy()
        fit = {'root_pos': np.tile([.2, -.1, .3], (len(motion), 1)),
               'root_quat_xyzw': np.tile([0., 0., 0., 1.], (len(motion), 1)),
               'joint_pos': np.zeros((len(motion), 27)), 'joint_names': tuple(self.model.actuated)}
        original = motion.copy()
        base = prepare_fitted_clip(motion, fit['joint_pos'], fit['joint_names'], self.model)
        corrected = prepare_root_fitted_clip(motion, fit, self.model)
        np.testing.assert_allclose(corrected['root_pos'], fit['root_pos'])
        np.testing.assert_allclose(corrected['points_z_up_world'] - base['points_z_up_world'],
                                   np.broadcast_to(fit['root_pos'][:, None, :], (len(motion), 22, 3)), atol=1e-12)
        np.testing.assert_array_equal(motion, original)
        fit['root_quat_xyzw'][:, 3] = 2
        with self.assertRaisesRegex(ValueError, 'normalized'):
            prepare_root_fitted_clip(motion, fit, self.model)


if __name__ == "__main__":
    unittest.main()

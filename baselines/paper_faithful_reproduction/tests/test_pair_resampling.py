"""Physical-clock regression tests; do not call these motion tracking results."""
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from screen_phygile_pairs_mjlab import resample_poses


class ClockTests(unittest.TestCase):
    def test_velocity_and_clock_preserved_with_unaligned_endpoint(self):
        n = np.zeros((4, 71)); n[:, 3] = 1
        t = np.arange(4)/20
        n[:, 0] = 2*t; n[:, 13] = 3*t
        p, q, j, c = resample_poses(n, 20., 50.)
        np.testing.assert_allclose(p[:, 0], np.arange(8)/50*2, atol=1e-12)
        np.testing.assert_allclose(j[:, 0], np.arange(8)/50*3, atol=1e-12)
        self.assertAlmostEqual(c['omitted_tail_s'], .01)
        self.assertFalse(c['time_stretched'])

    def test_rotation_crosses_pi_by_short_arc(self):
        n = np.zeros((3, 71))
        n[:, 3:7] = Rotation.from_euler('z', [170, 180, 190], degrees=True).as_quat()[:, [3, 0, 1, 2]]
        _, q, _, c = resample_poses(n, 20., 50.)
        actual = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        expected = Rotation.from_euler('z', 170+np.arange(6)*4, degrees=True).as_matrix()
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertEqual(c['omitted_tail_s'], 0.)

    def test_nonfinite_rejected(self):
        n = np.zeros((3, 71)); n[:, 3] = 1; n[1, 0] = np.nan
        with self.assertRaises(ValueError):
            resample_poses(n, 20., 50.)


if __name__ == '__main__':
    unittest.main()

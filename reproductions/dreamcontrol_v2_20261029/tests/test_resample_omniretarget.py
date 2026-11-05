import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from resample_omniretarget import resample, split_resampled


class ResampleTest(unittest.TestCase):
    def test_fractional_grid_and_duration_limit(self):
        q = np.zeros((361, 36))
        q[:, 0] = 1
        q[:, 4] = np.arange(len(q)) / 30
        q[:, 7] = 2 * q[:, 4]
        result = resample(q, 30)
        self.assertEqual(result.shape, (200, 36))
        np.testing.assert_allclose(result[:, 4], np.arange(200) / 20, atol=1e-12)
        np.testing.assert_allclose(result[:, 7], 2 * np.arange(200) / 20, atol=1e-12)
        np.testing.assert_allclose(result[:, 0], 1)

    def test_slerp_preserves_orientation_and_object_quaternion_sign(self):
        q = np.zeros((4, 43))
        angles = np.arange(4) * np.pi / 6
        q[:, 0] = np.cos(angles / 2)
        q[:, 3] = np.sin(angles / 2)
        q[:, 36] = [1, -1, 1, -1]
        result = resample(q, 30, target_fps=20)
        np.testing.assert_allclose(result[1, :4], [np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)], atol=1e-12)
        np.testing.assert_allclose(np.abs(result[:, 36]), 1, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(result[:, :4], axis=1), 1, atol=1e-12)

    def test_rejects_zero_quaternion(self):
        with self.assertRaisesRegex(ValueError, 'zero quaternion'):
            resample(np.zeros((30, 36)), 30)

    def test_rejects_nonunit_object_quaternion_instead_of_hiding_it(self):
        q = np.zeros((4, 43))
        q[:, 0] = 1
        q[:, 36] = .74
        with self.assertRaisesRegex(ValueError, 'not normalized'):
            resample(q, 30)

    def test_full_sequence_split_preserves_every_frame(self):
        q = np.zeros((760, 36))
        q[:, 0] = 1
        q[:, 4] = np.arange(len(q)) / 30
        full = resample(q, 30, max_seconds=None)
        clips, dropped = split_resampled(q, 30)
        self.assertEqual([len(x) for x in clips], [200, 200, 107])
        self.assertEqual(dropped, 0)
        np.testing.assert_allclose(np.concatenate(clips), full)

    def test_single_frame_tail_is_reported(self):
        q = np.zeros((301, 36))
        q[:, 0] = 1
        clips, dropped = split_resampled(q, 30)
        self.assertEqual([len(x) for x in clips], [200])
        self.assertEqual(dropped, 1)


if __name__ == '__main__':
    unittest.main()

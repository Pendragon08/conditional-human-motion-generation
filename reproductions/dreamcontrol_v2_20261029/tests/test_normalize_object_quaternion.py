import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from normalize_object_quaternion import normalize_object_quaternions


class QuaternionRepairTest(unittest.TestCase):
    def test_repair_is_explicit_and_preserves_other_fields(self):
        q = np.zeros((3, 43)); q[:, 0] = 1; q[:, 36:40] = [.5, .5, 0, 0]
        q[:, 7:36] = np.arange(29)
        original = q.copy()
        repaired, metadata = normalize_object_quaternions(q)
        np.testing.assert_array_equal(q, original)
        np.testing.assert_array_equal(repaired[:, :36], original[:, :36])
        np.testing.assert_array_equal(repaired[:, 40:], original[:, 40:])
        np.testing.assert_allclose(np.linalg.norm(repaired[:, 36:40], axis=1), 1.)
        np.testing.assert_allclose(repaired[:, 36:40], original[:, 36:40] / np.sqrt(.5))
        self.assertEqual(metadata['frames_outside_tolerance'], 3)
        self.assertEqual(metadata['frames_normalized'], 3)

    def test_rejects_zero_or_unreasonable_quaternion(self):
        q = np.zeros((2, 43)); q[:, 0] = 1
        with self.assertRaises(ValueError):
            normalize_object_quaternions(q)
        q[:, 36] = 3
        with self.assertRaises(ValueError):
            normalize_object_quaternions(q)


if __name__ == '__main__':
    unittest.main()

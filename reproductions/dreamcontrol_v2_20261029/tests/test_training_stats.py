import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from training_stats import training_stats


class TrainingStatsTest(unittest.TestCase):
    def test_matches_direct_calculation_without_validation_leakage(self):
        a = np.array([[1., 2., 7.], [3., 4., 7.]])
        b = np.array([[5., 6., 7.]])
        validation = np.full((10, 3), 1e12)
        result = training_stats([('train', a), ('val', validation), ('train', b)], feature_dim=3)
        expected = np.concatenate([a, b])
        np.testing.assert_allclose(result['mean'], expected.mean(axis=0))
        np.testing.assert_allclose(result['raw_std'], expected.std(axis=0))
        self.assertEqual(result['frame_count'], 3)
        self.assertEqual(result['clip_count'], 2)
        self.assertEqual(result['clamped_channels'], [2])
        self.assertEqual(result['std'][2], 1e-6)

    def test_rejects_bad_training_data_and_unknown_splits(self):
        with self.assertRaises(ValueError):
            training_stats([('train', np.array([[np.nan]]))], feature_dim=1)
        with self.assertRaises(ValueError):
            training_stats([('invalid', np.ones((2, 1)))], feature_dim=1)
        with self.assertRaises(ValueError):
            training_stats([('val', np.ones((2, 1)))], feature_dim=1)


if __name__ == '__main__':
    unittest.main()

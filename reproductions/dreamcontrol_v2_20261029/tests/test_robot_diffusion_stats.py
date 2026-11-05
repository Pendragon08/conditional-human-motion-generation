import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import torch
    from robot_diffusion_stats import configure_robot_stats
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'requires torch')
class RobotStatsTest(unittest.TestCase):
    def test_sets_robot_stats_and_handles_variable_length_and_empty_mask(self):
        diffusion = SimpleNamespace()
        configure_robot_stats(diffusion, np.arange(263), np.ones(263), np.zeros((22, 3)), np.ones((22, 3))*2)
        self.assertEqual(tuple(diffusion.mean.shape), (263,))
        self.assertEqual(tuple(diffusion.raw_std.shape), (66,))
        mask = torch.zeros((2, 200, 22, 1), dtype=torch.bool)
        mask[0, :5, 20, 0] = True
        self.assertEqual(tuple(diffusion.calc_grad_scale(mask).shape), (2, 1, 1, 1))
        np.testing.assert_allclose(diffusion.calc_grad_scale(mask).numpy().reshape(-1), [4., 0.])

    def test_invalid_stats_do_not_mutate_existing_object(self):
        diffusion = SimpleNamespace(mean='unchanged')
        with self.assertRaises(ValueError):
            configure_robot_stats(diffusion, np.zeros(263), np.zeros(263), np.zeros(66), np.ones(66))
        self.assertEqual(diffusion.mean, 'unchanged')


if __name__ == '__main__':
    unittest.main()

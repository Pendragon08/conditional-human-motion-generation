import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import torch
    from robot_spatial_guidance import guide
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'requires torch and DreamControl data loaders')
class SpatialGuidanceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.diffusion = SimpleNamespace(mean=torch.zeros(263), std=torch.ones(263),
                                         raw_mean=torch.zeros(66), raw_std=torch.ones(66),
                                         posterior_log_variance_clipped=np.log(np.full(1000, .2)))
        self.x = torch.randn(2, 263, 1, 20) * .05
        self.hint = torch.zeros(2, 20, 66)
        self.mask = torch.zeros(2, 20, 22, dtype=torch.bool)
        self.mask[0, 10, 20] = True
        self.times = torch.tensor([200, 700])

    def test_zero_target_is_active_and_empty_example_is_unchanged(self):
        out = guide(self.diffusion, self.x, self.times, self.hint, self.mask, step_size=1., steps=5)
        before = recover_from_ric(self.x.permute(0, 3, 2, 1).squeeze(2), 22)[0, 10, 20].square().sum()
        after = recover_from_ric(out.permute(0, 3, 2, 1).squeeze(2), 22)[0, 10, 20].square().sum()
        self.assertLess(float(after), float(before))
        torch.testing.assert_close(out[1], self.x[1])
        self.assertTrue(torch.isfinite(out).all())

    def test_padding_masks_constraints_and_bad_mask_is_rejected(self):
        valid = torch.ones((2, 20), dtype=torch.bool); valid[0, 10] = False
        out = guide(self.diffusion, self.x, self.times, self.hint, self.mask, step_size=1., steps=2, valid_frames=valid)
        torch.testing.assert_close(out, self.x)
        with self.assertRaises(ValueError):
            guide(self.diffusion, self.x, self.times, self.hint, self.mask.float(), step_size=1., steps=2)

    def test_distinct_batch_timesteps_use_distinct_variances(self):
        self.x[1] = self.x[0]
        self.mask[1] = self.mask[0]
        self.diffusion.posterior_log_variance_clipped[200] = np.log(.01)
        self.diffusion.posterior_log_variance_clipped[700] = np.log(.5)
        out = guide(self.diffusion, self.x, self.times, self.hint, self.mask,
                    step_size=1., steps=1, min_variance=0.)
        first = torch.linalg.vector_norm(out[0]-self.x[0])
        second = torch.linalg.vector_norm(out[1]-self.x[1])
        self.assertGreater(float(second), float(first) * 40.)


if __name__ == '__main__':
    unittest.main()

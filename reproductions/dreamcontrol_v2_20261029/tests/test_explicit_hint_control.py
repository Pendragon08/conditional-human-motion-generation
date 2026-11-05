import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import torch
    from explicit_hint_control import attach_explicit_hint_mask
except ImportError:
    torch = None


if torch is not None:
    class Hint(torch.nn.Module):
        def forward(self, x):
            return x[:, :, :3].permute(1, 0, 2)

    class TimeEmbedding(torch.nn.Module):
        def forward(self, t):
            return torch.zeros((1, len(t), 3), dtype=torch.float32, device=t.device)

    class Input(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[-1], x.shape[0], 3), device=x.device)

    class Encoder(torch.nn.Module):
        def forward(self, x):
            return x.unsqueeze(0)

    class FakeCMDM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.))
            self.latent_dim = 3
            self.cond_mode = 'no_cond'
            self.input_hint_block = Hint()
            self.c_embed_timestep = TimeEmbedding()
            self.c_input_process = Input()
            self.c_sequence_pos_encoder = torch.nn.Identity()
            self.c_seqTransEncoder = Encoder()
            self.zero_convs = torch.nn.ModuleList([torch.nn.Identity()])

        def cmdm_forward(self, x, t, y=None, weight=1.):
            raise NotImplementedError


@unittest.skipIf(torch is None, 'requires torch')
class ExplicitMaskTest(unittest.TestCase):
    def test_cancelling_coordinates_are_still_active_and_masked_data_are_ignored(self):
        model = FakeCMDM(); attach_explicit_hint_mask(model)
        x = torch.zeros((1, 263, 1, 2)); t = torch.tensor([20])
        hint = torch.zeros((1, 2, 66)); hint[0, 1, :3] = torch.tensor([1., -1., 0.])
        mask = torch.zeros((1, 2, 22), dtype=torch.bool); mask[0, 1, 0] = True
        control = model.cmdm_forward(x, t, {'hint': hint, 'hint_mask': mask})
        torch.testing.assert_close(control[0, 2, 0], torch.tensor([1., -1., 0.]))
        mask.zero_()
        torch.testing.assert_close(model.cmdm_forward(x, t, {'hint': hint, 'hint_mask': mask}), torch.zeros_like(control))

    def test_zero_target_mask_projection_has_gradients_and_rejects_bad_mask(self):
        model = FakeCMDM(); attach_explicit_hint_mask(model)
        x = torch.zeros((1, 263, 1, 2)); t = torch.tensor([20]); hint = torch.zeros((1, 2, 66))
        mask = torch.zeros((1, 2, 22), dtype=torch.bool); mask[0, 1, 0] = True
        model.cmdm_forward(x, t, {'hint': hint, 'hint_mask': mask}).sum().backward()
        self.assertGreater(float(model.hint_mask_embedding.weight.grad.abs().sum()), 0.)
        with self.assertRaises(ValueError):
            model.cmdm_forward(x, t, {'hint': hint, 'hint_mask': mask.float()})
        with self.assertRaises(ValueError):
            attach_explicit_hint_mask(model)


if __name__ == '__main__':
    unittest.main()

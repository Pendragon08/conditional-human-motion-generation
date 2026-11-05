import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from robot_training_step import train_step


class DummyDiffusion:
    model_mean_type = SimpleNamespace(name='START_X')
    posterior_log_variance_clipped = [0.] * 10
    def q_sample(self, clean, t, noise):
        return clean + noise


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.5))
    def forward(self, noisy, timesteps, y):
        return noisy * self.weight


class TrainingStepTests(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=.01)
        self.clean = torch.ones(2,263,1,3)
        self.cond = {'y':{'mask':torch.ones(2,1,1,3,dtype=torch.bool), 'hint':torch.zeros(2,3,66),
                          'hint_mask':torch.zeros(2,3,22,dtype=torch.bool), 'text':['first','second']}}
        self.kwargs = {'guidance_steps':0,'guidance_scale':.5,'timesteps':torch.tensor([1,8]),
                       'noise':torch.zeros_like(self.clean)}

    def test_optimizes_clean_motion_prediction(self):
        before = self.model.weight.item()
        result = train_step(self.model,DummyDiffusion(),self.optim,self.clean,self.cond,**self.kwargs)
        self.assertAlmostEqual(result['loss'], .25)
        self.assertGreater(self.model.weight.item(), before)
        self.assertEqual(result['timesteps'], [1,8])

    def test_guidance_receives_explicit_mask(self):
        self.kwargs['guidance_steps'] = 2
        with patch('robot_training_step.guide', side_effect=lambda d,x,t,h,m,**kw: x) as mock:
            train_step(self.model,DummyDiffusion(),self.optim,self.clean,self.cond,**self.kwargs)
        self.assertEqual(mock.call_args.args[4].shape, (2,3,22))
        self.assertEqual(mock.call_args.kwargs['steps'], 2)

    def test_rejects_padded_batch_without_updating(self):
        self.cond['y']['mask'][0,0,0,-1] = False
        before = self.model.weight.item()
        with self.assertRaisesRegex(ValueError, 'padded'):
            train_step(self.model,DummyDiffusion(),self.optim,self.clean,self.cond,**self.kwargs)
        self.assertEqual(before,self.model.weight.item())

    def test_rejects_wrong_diffusion_target(self):
        diffusion = DummyDiffusion()
        diffusion.model_mean_type = SimpleNamespace(name='EPSILON')
        with self.assertRaisesRegex(ValueError, 'clean-motion'):
            train_step(self.model,diffusion,self.optim,self.clean,self.cond,**self.kwargs)


if __name__ == '__main__':
    unittest.main()

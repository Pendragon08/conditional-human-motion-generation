import unittest
from types import SimpleNamespace
import torch
from robot_training_step import train_accumulated_step


class Diffusion:
    model_mean_type=SimpleNamespace(name='START_X')
    posterior_log_variance_clipped=[0.]*4
    def q_sample(self, clean, timesteps, noise): return clean


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(.5))
    def forward(self, x, t, y): return self.weight*x


def batch(n,t,value):
    x=torch.full((n,263,1,t),value)
    return x, {'y':{'mask':torch.ones(n,1,1,t,dtype=torch.bool),
                    'hint':torch.zeros(n,t,66),'hint_mask':torch.zeros(n,t,22,dtype=torch.bool),
                    'text':['x']*n}}


class AccumulationTests(unittest.TestCase):
    def test_equal_length_matches_single_batch(self):
        accumulated=Model(); combined=Model()
        opt_a=torch.optim.SGD(accumulated.parameters(),lr=.1)
        opt_b=torch.optim.SGD(combined.parameters(),lr=.1)
        x1,c1=batch(2,4,1.); x2,c2=batch(1,4,2.)
        train_accumulated_step(accumulated,Diffusion(),opt_a,[(x1,c1),(x2,c2)],
                               guidance_steps=0,guidance_scale=0.)
        x=torch.cat([x1,x2]); opt_b.zero_grad()
        ((combined.weight*x-x)**2).mean().backward(); opt_b.step()
        self.assertTrue(torch.allclose(accumulated.weight,combined.weight,atol=1e-7,rtol=0))

    def test_unequal_lengths_and_sizes_use_per_sample_mean(self):
        model=Model(); opt=torch.optim.SGD(model.parameters(),lr=.1)
        result=train_accumulated_step(model,Diffusion(),opt,[batch(2,4,1.),batch(1,7,2.)],
                                      guidance_steps=0,guidance_scale=0.)
        # Mean loss = (2*.25 + 1*1)/3; derivative at weight=.5 = -2.
        self.assertAlmostEqual(result['loss'],.5,places=6)
        self.assertAlmostEqual(model.weight.item(),.7,places=6)
        self.assertEqual(result['batch_size'],3)
        self.assertEqual([m['frames'] for m in result['microbatches']],[4,7])

    def test_invalid_second_batch_never_updates(self):
        model=Model(); opt=torch.optim.SGD(model.parameters(),lr=.1)
        bad=batch(1,7,2.); bad[1]['y']['mask'][0,0,0,0]=False
        with self.assertRaisesRegex(ValueError,'padded'):
            train_accumulated_step(model,Diffusion(),opt,[batch(1,4,1.),bad],
                                   guidance_steps=0,guidance_scale=0.)
        self.assertEqual(model.weight.item(),.5)
        self.assertIsNone(model.weight.grad)


if __name__=='__main__': unittest.main()

import unittest
from unittest.mock import patch
import numpy as np
import torch
from diffusion import gaussian_diffusion as gd
from robot_diffusion_stats import configure_robot_stats
from robot_prior_sampling import sample_robot_prior


class SmallConditionalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor=torch.nn.Parameter(torch.zeros(()))
        self.hint_mask_embedding=torch.nn.Linear(22,1,bias=False)
        self.cond_mask_prob=.1;self.rot2xyz=None;self.translation=True
        self.njoints=263;self.nfeats=1;self.data_rep='hml_vec';self.cond_mode='both_text_spatial'
        self.fail=False
    def forward(self,x,timesteps,y=None):
        if self.fail:raise RuntimeError('synthetic model failure')
        return torch.zeros_like(x) + (0. if y.get('uncond',False) else 1.) + self.anchor*0


def diffusion():
    def array(path):
        n=66 if 'raw' in str(path).lower() else 263
        return np.ones(n,dtype=np.float32) if 'Std' in str(path) else np.zeros(n,dtype=np.float32)
    with patch.object(gd.np,'load',side_effect=array):
        d=gd.GaussianDiffusion(betas=np.linspace(.001,.02,4),model_mean_type=gd.ModelMeanType.START_X,
            model_var_type=gd.ModelVarType.FIXED_SMALL,loss_type=gd.LossType.MSE,dataset='humanml')
    configure_robot_stats(d,np.zeros(263),np.ones(263),np.zeros(66),np.ones(66))
    return d


class RobotSamplingTest(unittest.TestCase):
    def setUp(self):
        self.model=SmallConditionalModel().eval();self.diff=diffusion()
        self.target=torch.zeros((2,6,22,3),dtype=torch.float32)
        self.mask=torch.zeros((2,6,22),dtype=torch.bool)
    def run_sample(self,*,cfg=1.,steps=0,seed=17,targets=None,mask=None):
        return sample_robot_prior(self.model,self.diff,texts=['one','two'],
            targets=self.target if targets is None else targets,target_mask=self.mask if mask is None else mask,
            cfg_scale=cfg,spatial_step_size=.5,spatial_steps=steps,seed=seed)

    def test_real_ddpm_cfg_and_reproducible_rng(self):
        before=torch.random.get_rng_state().clone()
        a=self.run_sample(cfg=0.);b=self.run_sample(cfg=2.)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertTrue(torch.allclose(a['normalized_features'],torch.zeros_like(a['normalized_features']),atol=1e-6))
        self.assertTrue(torch.allclose(b['normalized_features'],torch.full_like(b['normalized_features'],2.),atol=1e-6))
        self.assertEqual(tuple(b['canonical_points'].shape),(2,6,22,3))
        self.assertTrue(torch.equal(b['normalized_features'],self.run_sample(cfg=2.)['normalized_features']))
        self.assertFalse(b['metadata']['training_or_physical_acceptance'])

    def test_explicit_zero_normalized_constraint_is_not_dropped(self):
        self.diff.raw_mean=torch.ones(66);self.diff.raw_std=torch.ones(66)
        target=torch.ones_like(self.target);mask=self.mask.clone();mask[:,3,1]=True
        unconstrained=self.run_sample(cfg=0.,steps=1,targets=target)
        constrained=self.run_sample(cfg=0.,steps=1,targets=target,mask=mask)
        self.assertFalse(torch.equal(unconstrained['normalized_features'],constrained['normalized_features']))
        self.assertNotIn('guide',self.diff.__dict__)

    def test_restore_and_validate(self):
        self.model.fail=True
        with self.assertRaisesRegex(RuntimeError,'synthetic model failure'):self.run_sample()
        self.assertNotIn('guide',self.diff.__dict__)
        self.assertNotIn('_robot_sampling_active',self.diff.__dict__)
        self.model.fail=False
        with self.assertRaisesRegex(ValueError,'target mask'):
            self.run_sample(mask=self.mask.float())
        with self.assertRaisesRegex(ValueError,'share a device and use float32'):
            self.run_sample(targets=self.target.double())


if __name__=='__main__':unittest.main()

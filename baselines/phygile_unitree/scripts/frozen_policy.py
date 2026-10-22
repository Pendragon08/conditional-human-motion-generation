"""Load the original PhyGile MoE and empirical statistics without Isaac imports.

No policy architecture conversion and no PPO optimizer are created on this path.
The normalizer formula is checked against a saved original-runtime fixture.
"""
import importlib.util,sys
from pathlib import Path
import torch

class FrozenPhyGile:
    def __init__(self,checkpoint,contract,encoder_source,device='cpu'):
        name='phygile_frozen_encoder'
        spec=importlib.util.spec_from_file_location(name,encoder_source)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        payload=torch.load(checkpoint,map_location=device,weights_only=False)
        cfg=dict(contract['policy_config']);cfg.pop('class_name')
        self.model=module.MoEActorCritic(num_actor_obs=contract['observation_dims']['policy'][0],
          num_critic_obs=contract['observation_dims']['critic'][0],num_actions=len(contract['joint_names']),**cfg).to(device)
        if 'artifact_kind' in payload:
            from active_policy_artifact import load_active
            load_active(self.model,payload,contract)
        else:self.model.load_state_dict(payload['model_state_dict'],strict=True)
        self.model._max_active_experts=contract['active_experts']
        self.model.eval();self.norm={k:v.to(device) for k,v in payload['obs_norm_state_dict'].items()}
        if not {'_mean','_std'}.issubset(self.norm):raise ValueError('Unknown empirical normalizer state schema')
        self.device=device
    @torch.inference_mode()
    def __call__(self,obs):
        # RSL-RL 2.3.1 EmpiricalNormalization inference formula, epsilon=1e-2.
        x=(obs.to(self.device)-self.norm['_mean'])/(self.norm['_std']+1e-2)
        result=self.model.act_inference(x)
        if not torch.isfinite(result).all():raise RuntimeError('Nonfinite frozen-policy action')
        return result

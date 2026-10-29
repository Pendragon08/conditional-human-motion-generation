"""Prepared CPU audit and actor+RMS payload core; no executable launch path.

Invoke only in a future authorized, bounded CPU supervisor after the genuine
endpoint. Importing this file itself imports no neural runtime or simulator.
"""
import inspect
import math
import os
from pathlib import Path
import random
from step4_ppo5000_frozen_protocol_v2 import sha,validate_fullstate_metadata

NATIVE_SHA={
 'actor':'51ce8c6504a7434f86e5188ccb11e9fc981e46f764c00e32be3e8febe7112ae2',
 'normalizer':'f52677617f729c21c7440fd52d69b7e570313836a9deb21519425b806043a311',
 'runner':'2f00052c1088f8d13f1b106f25981dc95ff081b2a23841413cca3a7896e0b262'}

def audit_and_payload(state,binding):
 """No optimizer update, CUDA-RNG application, SDK import, or action clipping."""
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 import numpy as np
 import torch
 from rsl_rl.modules.actor_critic import ActorCritic
 from rsl_rl.modules.normalizer import EmpiricalNormalization
 from step4_train_only_ppo_runtime_v4 import clone,equal,content
 assert not torch.cuda.is_initialized() and torch.cuda.device_count()==0
 torch.set_num_threads(4)
 for name,cls in [('actor',ActorCritic),('normalizer',EmpiricalNormalization)]:
  assert sha(inspect.getfile(cls))==NATIVE_SHA[name]
 # Verify runner source without constructing the environment/runner.
 native=Path(inspect.getfile(ActorCritic)).parent.parent
 assert sha(native/'runners/on_policy_runner.py')==NATIVE_SHA['runner']
 meta=validate_fullstate_metadata(state,binding)
 tensor_count=0
 def finite_cpu(value):
  nonlocal tensor_count
  if torch.is_tensor(value):
   assert value.device.type=='cpu' and torch.isfinite(value).all();tensor_count+=1
  elif isinstance(value,np.ndarray):assert np.isfinite(value).all()
  elif isinstance(value,(float,np.floating)):assert math.isfinite(float(value))
  elif isinstance(value,dict):
   for child in value.values():finite_cpu(child)
  elif isinstance(value,(list,tuple)):
   for child in value:finite_cpu(child)
 finite_cpu(state)
 original=content(state)
 rng=state['rng']
 random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy'])
 for key in ['cpu','cuda_device1']:
  x=rng[key];assert x.dtype==torch.uint8 and x.ndim==1 and x.numel()>0
 torch.Generator(device='cpu').set_state(rng['cpu'])
 # CUDA RNG bytes are inspected, never applied in this CPU process.
 noise='scalar' if 'std' in state['model'] else 'log'
 assert ('std' in state['model']) != ('log_std' in state['model'])
 model=ActorCritic(431,557,29,actor_hidden_dims=[2048,1024,512],
  critic_hidden_dims=[2048,1024,512],activation='elu',noise_std_type=noise).cpu()
 model.load_state_dict(state['model'],strict=True)
 assert equal(model.state_dict(),state['model'])
 params=list(model.parameters());opt=state['optimizer']
 ids=[i for group in opt['param_groups'] for i in group['params']]
 assert len(ids)==len(set(ids))==len(params)==len(opt['state'])
 for i,param in zip(ids,params):
  row=opt['state'][i]
  assert set(row)=={'step','exp_avg','exp_avg_sq'}
  assert row['step'].numel()==1 and float(row['step'])==100000
  assert row['exp_avg'].shape==row['exp_avg_sq'].shape==param.shape
  assert row['exp_avg'].dtype==row['exp_avg_sq'].dtype==param.dtype
 for group in opt['param_groups']:assert group['lr']==state['learning_rate']
 normalizers=[]
 for name,dim in [('obs_normalizer',431),('critic_normalizer',557)]:
  value=state[name]
  assert value['count'].dtype==torch.long and value['count'].numel()==1 and int(value['count'])>0
  for key in ['_mean','_std','_var']:assert value[key].shape==(1,dim) and value[key].dtype==torch.float32
  assert (value['_std']>=0).all() and (value['_var']>=0).all()
  norm=EmpiricalNormalization([dim],eps=1e-2,until=1e8)
  norm.load_state_dict(value,strict=True);assert equal(norm.state_dict(),value)
  norm.eval().requires_grad_(False);normalizers.append(norm)
 model.eval().requires_grad_(False)
 norm=normalizers[0]
 before=content(dict(model=model.state_dict(),normalizer=norm.state_dict()))
 probes=[torch.zeros((1,431),dtype=torch.float32),
  torch.linspace(-2,2,7*431,dtype=torch.float32).reshape(7,431)]
 rng_before=torch.get_rng_state().clone()
 with torch.inference_mode():expected=[model.act_inference(norm(x)).clone() for x in probes]
 assert all(x.shape==(len(p),29) and torch.isfinite(x).all() for x,p in zip(expected,probes))
 assert any(bool((x!=0).any()) for x in expected)
 assert torch.equal(rng_before,torch.get_rng_state())
 assert content(dict(model=model.state_dict(),normalizer=norm.state_dict()))==before
 # The payload contains only the deterministic actor and its learned RMS.
 # Critic/noise/Adam/ADS/RNG remain in the private fullstate, not deployment.
 payload=dict(schema='textop.original_train_ppo5000_actor_RMS.v1',training_binding=binding,
  actor=clone(model.actor.state_dict()),obs_normalizer=clone(norm.state_dict()),
  actor_observation_dim=431,action_dim=29,hidden=[2048,1024,512],activation='elu',
  normalization=dict(kind='EmpiricalNormalization',eps=.01,until=1e8,eval_mode=True),
  input='raw assembled431D policy observation before RMS; no critic terms',
  output='raw deterministic29D policy action before environment actuator scaling',
  policy_hz=50,future_reference_frames=5,action_clamp_added=False,
  native_sources=NATIVE_SHA,heldout_tracking_accepted=False,online_latency_accepted=False,
  physics_trajectory_replay_claimed=False)
 assert content(state)==original
 assert not torch.cuda.is_initialized()
 return payload,dict(fullstate_metadata=meta,finite_cpu_tensors=tensor_count,
  full_model_exact=True,full_Adam_tensor_shape_step_and_finite_audit=True,
  CPU_optimizer_step_performed=False,CUDA_RNG_restored=False,
  actor_RMS_probe_output_hashes=[content(x) for x in expected],
  fullstate_content_sha256=original,payload_content_sha256=content(payload)),expected,probes

def independent_reload(payload,expected,probes):
 """Reload actor and learned RMS without a critic, then compare exact outputs."""
 import torch
 from rsl_rl.modules.normalizer import EmpiricalNormalization
 from step4_train_only_ppo_runtime_v4 import content,equal
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
 assert payload['schema']=='textop.original_train_ppo5000_actor_RMS.v1'
 assert (payload['actor_observation_dim'],payload['action_dim'],payload['policy_hz'],payload['future_reference_frames'])==(431,29,50,5)
 assert payload['hidden']==[2048,1024,512] and payload['activation']=='elu'
 assert payload['native_sources']==NATIVE_SHA and payload['action_clamp_added'] is False
 assert payload['normalization']==dict(kind='EmpiricalNormalization',eps=.01,until=1e8,eval_mode=True)
 sizes=[431,2048,1024,512,29];layers=[]
 for index,(left,right) in enumerate(zip(sizes,sizes[1:])):
  layers.append(torch.nn.Linear(left,right))
  if index<3:layers.append(torch.nn.ELU())
 actor=torch.nn.Sequential(*layers)
 actor.load_state_dict(payload['actor'],strict=True)
 norm=EmpiricalNormalization([431],eps=.01,until=1e8)
 norm.load_state_dict(payload['obs_normalizer'],strict=True)
 actor.eval().requires_grad_(False);norm.eval().requires_grad_(False)
 before=content(dict(actor=actor.state_dict(),norm=norm.state_dict()))
 cpu_rng=torch.get_rng_state().clone()
 with torch.inference_mode():actual=[actor(norm(x)).clone() for x in probes]
 assert equal(actual,expected) and torch.equal(cpu_rng,torch.get_rng_state())
 assert content(dict(actor=actor.state_dict(),norm=norm.state_dict()))==before
 assert not torch.cuda.is_initialized()
 return dict(independent_actor_RMS_reload_exact=True,RMS_buffers_unchanged=True,
  sample_mean_actions=True,synthetic_probe_only=True,heldout_or_online_quality=False)

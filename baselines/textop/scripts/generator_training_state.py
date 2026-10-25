"""Exact primitive-step accounting and atomic state around official managers.

No optimizer/loss/sampling algorithm changes. Checkpoints are private trusted
local artifacts. Accepted-corpus gates remain the launcher's responsibility.
"""
import hashlib,json,pathlib,random,shutil
import numpy as np
import torch

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()

def equal(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().cpu(),b.detach().cpu())
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return type(a)==type(b) and a==b

def module_sha(module):
 h=hashlib.sha256()
 for name,value in module.state_dict().items():
  h.update(name.encode());h.update(str(value.dtype).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def rng_state(generators):
 return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
  'cuda':torch.cuda.get_rng_state_all(),'generators':{k:v.get_state() for k,v in generators.items()}}

class PrimitiveClock:
 def __init__(self,manager):
  self.manager=manager;self.updates=0;self.attempts=0;self.skipped=0;self.last_updates=0
  original=manager.optimizer.step
  def counted(*args,**kwargs):
   value=original(*args,**kwargs);self.updates+=1;return value
  manager.optimizer.step=counted

 def after_post_step(self):
  self.attempts+=1;delta=self.updates-self.last_updates
  assert delta in [0,1],'One manager primitive cannot contain multiple optimizer steps'
  self.skipped+=int(delta==0);self.last_updates=self.updates
  assert self.manager.step==self.attempts and self.attempts==self.updates+self.skipped

 def state(self):return {'actual_optimizer_updates':self.updates,'completed_primitive_attempts':self.attempts,'skipped_updates':self.skipped,'last_observed_updates':self.last_updates}

 def load(self,value):
  self.updates=value['actual_optimizer_updates'];self.attempts=value['completed_primitive_attempts']
  self.skipped=value['skipped_updates'];self.last_updates=value['last_observed_updates']
  assert self.last_updates==self.updates and self.attempts==self.updates+self.skipped==self.manager.step

def save_manager(manager,model_key,clock,generators,path,binding,minimum_free_bytes,budget_bytes):
 path=pathlib.Path(path);assert not path.exists() and not path.with_suffix('.partial').exists()
 assert manager.step%4==0,'Checkpoints must be at four-primitive batch boundaries'
 assert clock.attempts==manager.step and shutil.disk_usage(path.parent).free>minimum_free_bytes+budget_bytes
 state={model_key:getattr(manager,model_key).state_dict(),'optimizer':manager.optimizer.state_dict(),'step':manager.step,
  'ema_models':{k:v.state_dict() for k,v in manager.ema_models.items()},'primitive_clock':clock.state(),
  'rng':rng_state(generators),'binding':binding,'schema':'textop.generator_training_state.v1',
  'manager_fields':{'stage_idx':manager.stage_idx,'pending_evaluation_steps':manager._to_eval_steps,
   'eval_loss_totals':dict(manager._total_eval_loss_dict),'eval_extra_totals':dict(manager._total_eval_extras_dict),'extra':manager.extra},
  'checkpoint_boundary':'completed_four_primitive_batch; no cached batch/history'}
 partial=path.with_suffix('.partial');torch.save(state,partial);partial.replace(path)
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'manager_step':manager.step,'actual_optimizer_updates':clock.updates}

def load_manager(manager,model_key,clock,generators,path,binding,expected_sha):
 assert sha(path)==expected_sha
 state=torch.load(path,map_location='cpu')
 assert state['schema']=='textop.generator_training_state.v1' and state['binding']==binding
 assert state['step']%4==0 and state['binding']['feature_dim']==69 and state['binding']['dof']==29
 model=getattr(manager,model_key);model.load_state_dict(state[model_key],strict=True)
 manager.optimizer.load_state_dict(state['optimizer']);manager.step=state['step']
 assert manager.ema_models.keys()==state['ema_models'].keys()
 for key,value in state['ema_models'].items():manager.ema_models[key].load_state_dict(value,strict=True)
 fields=state['manager_fields'];manager.stage_idx=fields['stage_idx'];manager._to_eval_steps=fields['pending_evaluation_steps']
 manager._total_eval_loss_dict.clear();manager._total_eval_loss_dict.update(fields['eval_loss_totals'])
 manager._total_eval_extras_dict.clear();manager._total_eval_extras_dict.update(fields['eval_extra_totals']);manager.extra=fields['extra']
 clock.load(state['primitive_clock'])
 assert equal(model.state_dict(),state[model_key]) and equal(manager.optimizer.state_dict(),state['optimizer'])
 assert all(equal(manager.ema_models[k].state_dict(),v) for k,v in state['ema_models'].items())
 rng=state['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'].cpu())
 torch.cuda.set_rng_state_all([x.cpu() for x in rng['cuda']]);assert generators.keys()==rng['generators'].keys()
 for key,value in rng['generators'].items():generators[key].set_state(value.cpu())
 assert equal(rng_state(generators),rng)
 return {'model_restored_exactly':True,'optimizer_restored_exactly':True,'ema_restored_exactly':True,'rng_and_generator_restored_exactly':True,
  'clock_restored_exactly':True,'pending_evaluation_fields_restored':True,'checkpoint_boundary':state['checkpoint_boundary']}

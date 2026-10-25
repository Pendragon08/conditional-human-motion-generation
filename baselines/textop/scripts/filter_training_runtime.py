"""Recording/checkpoint layer over unchanged pinned PPO and motion sampling.

EMA sampling is not rewritten. Extra raw attempt counters exclude initial
resets and allow the evaluator to reject motions never actually attempted.
Motion ordering is identified by a fingerprint; no labels enter checkpoints.
"""
import hashlib,json,pathlib
import torch
from textop_tracker.tasks.tracking.mdp.commands_multi import MotionCommand

def fingerprint(rows):
 return hashlib.sha256(json.dumps([{'sha256':x['sha256'],'frames':x['frames']} for x in rows],sort_keys=True,separators=(',',':')).encode()).hexdigest()

class RecordedMotionCommand(MotionCommand):
 def __init__(self,cfg,env):
  super().__init__(cfg,env)
  self.raw_failed=torch.zeros(self.num_motion,dtype=torch.long,device=self.device)
  self.raw_success=torch.zeros_like(self.raw_failed)
  self._attempt_started=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
 def _resample_command(self,env_ids):
  prior=env_ids[self._attempt_started[env_ids]]
  if len(prior):
   failed=self._env.termination_manager.terminated[prior]
   self.raw_failed+=torch.bincount(self.motion_idx[prior][failed],minlength=self.num_motion)
   self.raw_success+=torch.bincount(self.motion_idx[prior][~failed],minlength=self.num_motion)
  super()._resample_command(env_ids)
  self._attempt_started[env_ids]=True
 def recording_state(self):
  names=['failed_motion_count','success_motion_count','raw_failed','raw_success']
  return {**{name:getattr(self,name).detach().cpu().clone() for name in names},
          'failed_total_count':float(self.failed_total_count),'success_total_count':float(self.success_total_count)}
 @torch.inference_mode()
 def restore_recording(self,state):
  for name in ['failed_motion_count','success_motion_count','raw_failed','raw_success']:
   value=getattr(self,name);saved=state[name]
   assert value.shape==saved.shape and value.dtype==saved.dtype
   value.copy_(saved.to(value.device))
  self.failed_total_count=state['failed_total_count'];self.success_total_count=state['success_total_count']
  self.current_failed_motion_count.zero_();self.current_success_motion_count.zero_()
  # New physics episodes have not completed an attempt from the saved run.
  self._attempt_started.zero_()

def equal_state(x,y):
 if torch.is_tensor(x):return x.dtype==y.dtype and x.shape==y.shape and torch.equal(x.detach().cpu(),y.detach().cpu())
 if isinstance(x,dict):return x.keys()==y.keys() and all(equal_state(x[k],y[k]) for k in x)
 if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equal_state(a,b) for a,b in zip(x,y))
 return x==y

def save_recorded(runner,motion,path,completed,updates,corpus_fingerprint,diagnostic_only):
 """Atomic checkpoint, completed-count semantics, independent of RSL loop index."""
 path=pathlib.Path(path);assert not path.exists()
 original=runner.current_learning_iteration;runner.current_learning_iteration=completed
 partial=path.with_name(path.name+'.partial')
 try:
  runner.save(str(partial),infos={'completed_learning_iterations':completed,'actual_optimizer_updates':updates,
       'reference_fingerprint':corpus_fingerprint,'diagnostic_only':diagnostic_only,'recording':motion.recording_state()})
  partial.replace(path)
 finally:runner.current_learning_iteration=original
 return {'name':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'completed_learning_iterations':completed,'actual_optimizer_updates':updates}

def load_recorded(runner,motion,path,corpus_fingerprint):
 saved=torch.load(path,map_location=runner.device,weights_only=False)
 info=saved['infos'];assert info['reference_fingerprint']==corpus_fingerprint
 runner.load(str(path),load_optimizer=True);motion.restore_recording(info['recording'])
 assert runner.current_learning_iteration==info['completed_learning_iterations']
 for actual,key in [(runner.alg.policy.state_dict(),'model_state_dict'),(runner.alg.optimizer.state_dict(),'optimizer_state_dict'),
                    (runner.obs_normalizer.state_dict(),'obs_norm_state_dict'),(runner.privileged_obs_normalizer.state_dict(),'privileged_obs_norm_state_dict')]:
  assert equal_state(actual,saved[key]),key
 assert equal_state(motion.recording_state(),info['recording'])
 return info

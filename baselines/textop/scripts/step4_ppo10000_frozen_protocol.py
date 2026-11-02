"""Disabled successor provenance for the genuine original-TRAIN 10000 endpoint.

Standard library only. Never turns a live count, emergency or old teacher into
a completed endpoint. This module does not acquire, release or signal a model.
"""
import hashlib
import json
import math
import os
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RUN='user_step4_train_only_ppo10000_train_v4'
PREFIX='step4_train_only_'+RUN
REQUEST=ROOT/'receipts'/(PREFIX+'_request.json')
REQUEST_SHA='715feef32953cc79491781753587aa0041dd154db12bfdaa8576d3f57de4aa82'
ENTRY_SHA='78d98f457e0a286263196ae523476895b71b56ae690a9c8ee889426805cf7bc8'
SUP_SHA='a15cf3e6eda8600f1cf95f439b79f4b263e93f7be3af287916b41e186238beaa'
RUNTIME_SHA='d6cec5d3c54bd30a8c2d9de98d855c759192b8e489056bf86c7068e62db8eecc'
LIBRARY_SHA='375b16c3b8d919561e59cea8830ab071f2a128f39bf76363474cf0c37eb5ddde'
RESERVE=2*1024**3
LOG=128*1024**2

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()

def owned(path):
 p=Path(path)
 assert p.is_absolute() and p.resolve()==p and (p==ROOT or ROOT in p.parents)
 assert not p.is_symlink() and all(not a.is_symlink() for a in p.parents)
 assert p.stat().st_uid==1013
 return p

def read(path):return json.loads(owned(path).read_bytes())

def fingerprint(path):
 p=owned(path);s=p.stat()
 return dict(path=str(p),device=s.st_dev,inode=s.st_ino,bytes=s.st_size,
  uid=s.st_uid,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,sha256=sha(p))

def validate_endpoint_receipt(rec):
 assert rec['run']==RUN and rec['role']=='train'
 assert rec['status']=='STAGE4_ORIGINAL_TRAIN_ONE_STAGE_PPO_FINISHED'
 assert rec['source_sha256']==ENTRY_SHA and rec['request_sha256']==REQUEST_SHA
 for key,wanted in [('formal_completed_iterations',10000),('formal_optimizer_updates',200000),
  ('optimizer_attempts_in_process',200000),('actual_optimizer_calls_in_process',200000),
  ('diagnostic_optimizer_calls',0),('old_teacher_training_updates',0),('VAE_updates',0),('LDM_updates',0)]:
  assert type(rec[key]) is int and rec[key]==wanted,key
 assert rec['output_writers_closed'] is True and rec['inputs_unchanged'] is True
 assert rec['training_data_mode']=='original_train_only' and rec['stage3_status']=='POSTPONED_BY_USER'
 assert rec['original_VAL_motions_in_training']==rec['generated_training_clips']==0
 assert rec['original_train_families']==5361 and rec['actual_original_train_clips']==5551
 assert (rec['actor_observation_dim'],rec['critic_observation_dim'],rec['action_dim'],rec['policy_hz'],rec['num_envs'])==(431,557,29,50,8192)
 assert rec['initial_optimizer_state_entries']==0
 assert rec['input_library_binding']['receipt_sha256']==LIBRARY_SHA
 assert 'emergency_last_good' not in rec
 cp=rec['checkpoints'];assert len(cp)==1
 item=cp[0]
 assert Path(item['path'])==ROOT/'step34_numeric_v1'/RUN/'completed_10000.private.pth'
 assert type(item['bytes']) is int and 0<item['bytes']<=128*1024**2
 for key in ['sha256','content_sha256']:
  assert len(item[key])==64 and all(c in '0123456789abcdef' for c in item[key])
 return item

def accepted_training(binding=None):
 assert os.getuid()==1013 and sha(owned(REQUEST))==REQUEST_SHA
 req=read(REQUEST);policy=req['policy']
 assert req['run']==RUN and req['role']=='train' and req['author_checkpoint'] is None
 assert policy['training_authorized'] is True and policy['witnesses_authorized'] is False
 assert policy['target_RL_iterations']==policy['human_selected_target_iterations']==10000
 assert policy['real_RL_witnesses_accepted'] is True
 assert req['executor_sha256']==ENTRY_SHA and req['supervisor_sha256']==SUP_SHA
 for name,h in req['bound_sources'].items():assert sha(owned(ROOT/name))==h
 out=owned(ROOT/'step34_numeric_v1'/RUN)
 sup=owned(out.with_name(RUN+'_supervision'))
 paths=dict(receipt=out/'receipt.json',exit=sup/'exit.json',process=sup/'process_exit.json',
  outer=ROOT/'receipts'/(PREFIX+'_outer_exit.json'))
 files={k:read(p) for k,p in paths.items()};hashes={k:sha(p) for k,p in paths.items()}
 rec=files['receipt'];item=validate_endpoint_receipt(rec);ex=files['exit']
 assert ex['source_sha256']==SUP_SHA and ex['request_sha256']==REQUEST_SHA
 assert ex['receipt_sha256']==hashes['receipt'] and ex['process_exit_sha256']==hashes['process']
 assert ex['resource_sha256']==sha(owned(sup/'resources.jsonl'))
 for key in ['exit','process']:
  row=files[key];assert row['child_exit_code']==0 and row['child_reaped'] is True
  assert row['sample_errors']==[] and row['timed_out'] is False
 assert ex['supervision_passed'] is True and files['outer']['actual_returncode']==0
 last=json.loads((sup/'resources.jsonl').read_text().splitlines()[-1])
 assert last['phase']=='after_exit' and last['owned_group_processes']==last['owned_group_gpu_memory']==[]
 # The future CPU supervisor holds the common lock itself. Inspect the old
 # training argv, never incorrectly require that lock to be free inside a job.
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:
   if proc.stat().st_uid!=1013:continue
   argv=(proc/'cmdline').read_bytes().decode().split('\0')
   assert not (str(REQUEST) in argv and any(str(ROOT/'scripts'/name) in argv for name in
    ['run_step4_train_only_ppo_v4.py','supervise_step4_train_only_ppo_v4.py'])),'Original worker remains'
  except (FileNotFoundError,ProcessLookupError):pass
 path=owned(item['path']);assert path.parent==out and path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
 actual=dict(kind='ORIGINAL_TRAIN_FRESH_PPO10000_ACTUAL_ENDPOINT',run=RUN,request_sha256=REQUEST_SHA,
  closure_sha256=hashes,input_library_binding=rec['input_library_binding'],checkpoint=item,
  executor_sha256=ENTRY_SHA,runtime_sha256=RUNTIME_SHA,supervisor_sha256=SUP_SHA)
 if binding is not None:assert binding==actual
 return dict(binding=actual,request=req,receipt=rec)

def validate_fullstate_metadata(state,binding):
 assert state['schema']=='textop.stage4_train_only_fullstate.v1'
 assert type(state['completed_iterations']) is int and state['completed_iterations']==10000
 assert type(state['optimizer_updates']) is int and state['optimizer_updates']==200000
 assert state['input_binding']==binding['input_library_binding']
 assert state['physics_continuation']=='new episodes; simulator caches/in-flight episodes not serialized'
 assert state['algorithm']=='fresh one-stage PPO; no teacher/DAgger/distillation'
 assert set(state['rng'])=={'python','numpy','cpu','cuda_device1'}
 assert set(state['runner_clock'])=={'tot_time','tot_timesteps'}
 clock=state['runner_clock'];assert math.isfinite(clock['tot_time']) and clock['tot_time']>0
 assert type(clock['tot_timesteps']) is int and clock['tot_timesteps']==10000*24*8192
 assert math.isfinite(state['learning_rate']) and state['learning_rate']>0
 assert set(state['adaptive_sampling'])=={'failed_motion_count','success_motion_count',
  'current_failed_motion_count','current_success_motion_count','failed_total_count',
  'success_total_count','raw_failed','raw_success'}
 assert set(state['obs_normalizer'])==set(state['critic_normalizer'])=={'_mean','_var','_std','count'}
 assert isinstance(state['model'],dict) and state['model']
 assert set(state['optimizer'])=={'state','param_groups'} and state['optimizer']['state']
 return dict(completed_iterations=10000,Adam_updates=200000,CUDA_RNG_restored=False,
  physics_trajectory_replay_claimed=False)

"""Provenance gates for the user-requested interrupted PPO5000 endpoint.

The controller's receipt proves a signal and file, not the file's actual step.
No tensors, model, SDK or CUDA are imported here. Only a later CPU model audit
can accept the selected file after verifying its fullstate contents.
"""
import hashlib,json,math,os,pathlib
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RUN='user_step4_train_only_ppo10000_train_v4'
CONTROL='user_ppo5000_control_v1'
REQUEST=ROOT/'receipts'/('step4_train_only_'+RUN+'_request.json')
CONTROL_SPEC=ROOT/'configs/ppo5000-user-control-spec.json'
REQUEST_SHA='715feef32953cc79491781753587aa0041dd154db12bfdaa8576d3f57de4aa82'
CONTROL_SPEC_SHA='4d446d9d6b34e5518f9f85d11090e4ac1482008b948c8ee930d5123d7fcfcc52'
ENTRY_SHA='78d98f457e0a286263196ae523476895b71b56ae690a9c8ee889426805cf7bc8'
SUP_SHA='a15cf3e6eda8600f1cf95f439b79f4b263e93f7be3af287916b41e186238beaa'
RUNTIME_SHA='d6cec5d3c54bd30a8c2d9de98d855c759192b8e489056bf86c7068e62db8eecc'
WATCHER_SHA='8218f10715b019eb78784db1eb8c59b5a1c40ef4081d78c1eab2ccf77476e8ba'
CTRL_SUP_SHA='dc87b77ecebea51ee4e85fee6916f355d3737380ebd11e0eb5fc148bcb4d2b27'
LIBRARY_SHA='375b16c3b8d919561e59cea8830ab071f2a128f39bf76363474cf0c37eb5ddde'
RESERVE=2*1024**3
LOG=128*1024**2

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()

def owned(path):
 p=pathlib.Path(path)
 assert p.is_absolute() and p.resolve()==p and (p==ROOT or ROOT in p.parents)
 assert not p.is_symlink() and all(not a.is_symlink() for a in p.parents)
 assert p.stat().st_uid==1013
 return p

def read(path):return json.loads(owned(path).read_bytes())

def fingerprint(path):
 p=owned(path);before=p.stat();digest=sha(p);after=p.stat()
 keys=['st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_uid']
 assert all(getattr(before,k)==getattr(after,k) for k in keys)
 return dict(path=str(p),device=before.st_dev,inode=before.st_ino,bytes=before.st_size,
  uid=before.st_uid,mtime_ns=before.st_mtime_ns,ctime_ns=before.st_ctime_ns,sha256=digest)

def validate_signal_evidence(ctrl,request,intent,sent,before,producer):
 """Validate metadata only; do not claim actual fullstate step yet."""
 assert request['run']==RUN and request['role']=='train' and request['author_checkpoint'] is None
 p=request['policy'];assert p['training_authorized'] is True and p['witnesses_authorized'] is False
 assert p['target_RL_iterations']==p['human_selected_target_iterations']==10000
 assert p['real_RL_witnesses_accepted'] is True and p['training_data_mode']=='original_train_only'
 assert request['executor_sha256']==ENTRY_SHA and request['supervisor_sha256']==SUP_SHA
 assert request['bound_sources']['scripts/step4_train_only_ppo_runtime_v4.py']==RUNTIME_SHA
 assert ctrl['run']==CONTROL and ctrl['status']=='PPO5000_SIGNAL_AND_SAVED_STATE_AWAITING_CPU_AUDIT'
 assert ctrl['original_run']==RUN and ctrl['signals_sent']==1 and ctrl['target_iterations']==5000
 assert ctrl['target_optimizer_updates']==100000 and ctrl['output_writers_closed'] is True
 assert ctrl['original_nonzero_exits_preserved'] is True and ctrl['common_model_lock_released'] is True
 assert ctrl['selected_checkpoint_payload_audited'] is False and ctrl['model_loads']==0
 assert ctrl['request_sha256']==CONTROL_SPEC_SHA and ctrl['watcher_sha256']==WATCHER_SHA
 assert intent['status']=='ONE_BOUND_PIDFD_SIGINT_DISPATCH_INTENT' and intent['signal']==2
 assert intent['target_iterations']==intent['observed_iterations']==5000 and intent['observed_updates']==100000
 assert intent['checkpoint_not_yet_persisted'] is True
 identity=intent['process'];assert identity['pid']==479092 and identity['start_ticks']=='1446080144' and identity['uid']==1013
 assert sent['signal']=='SIGINT' and sent['signals_sent']==1 and sent['target_iterations']==5000
 assert before['run']==RUN and before['status']=='RUNNING_STAGE4_FRESH_ORIGINAL_TRAIN_PPO'
 assert before['request_sha256']==REQUEST_SHA and before['source_sha256']==ENTRY_SHA
 assert before['formal_completed_iterations']==5000 and before['formal_optimizer_updates']==100000
 assert before['optimizer_attempts_in_process']==before['actual_optimizer_calls_in_process']==100000
 assert before['input_library_binding']['receipt_sha256']==LIBRARY_SHA
 assert producer['run']==RUN and producer['status']=='STAGE4_ORIGINAL_TRAIN_PPO_FAILED_PARTIAL_RETAINED'
 assert producer['error']=="RuntimeError('Owned Stage4 PPO interrupted with signal 2')"
 assert producer['source_sha256']==ENTRY_SHA and producer['request_sha256']==REQUEST_SHA
 assert producer['training_data_mode']=='original_train_only' and producer['stage3_status']=='POSTPONED_BY_USER'
 assert producer['original_VAL_motions_in_training']==producer['generated_training_clips']==0
 assert producer['old_teacher_training_updates']==producer['VAE_updates']==producer['LDM_updates']==0
 assert producer['initial_optimizer_state_entries']==0
 assert producer['input_library_binding']==before['input_library_binding']
 assert producer['input_library_binding']['receipt_sha256']==LIBRARY_SHA
 assert 'emergency_save_error' not in producer and producer['checkpoints']==[]
 checkpoint=producer['emergency_last_good'];assert checkpoint==ctrl['saved_state']
 assert pathlib.Path(checkpoint['path'])==ROOT/'step34_numeric_v1'/RUN/'emergency_last_good.private.pth'
 assert type(checkpoint['bytes']) is int and 0<checkpoint['bytes']<=128*1024**2
 for key in ['sha256','content_sha256']:
  assert len(checkpoint[key])==64 and all(c in '0123456789abcdef' for c in checkpoint[key])
 return checkpoint

def verify_exit(proof,expect_success):
 assert proof['child_reaped'] is True and proof['timed_out'] is False and proof['sample_errors']==[]
 assert proof['supervisor_error'] is None and not proof.get('cleanup_error') and proof['shutdown_actions']==[]
 if expect_success:assert proof['child_exit_code']==0 and proof['supervision_passed'] is True
 else:assert proof['child_exit_code'] in [1,-2,130] and proof['supervision_passed'] is False

def accepted_stop(binding=None):
 """Require real original and controller closure; target state is still unaudited."""
 assert os.getuid()==1013 and sha(owned(REQUEST))==REQUEST_SHA and sha(owned(CONTROL_SPEC))==CONTROL_SPEC_SHA
 request=read(REQUEST);spec=read(CONTROL_SPEC)
 assert spec['authorized'] is True and spec['run']==CONTROL and spec['target_iterations']==5000
 assert spec['target_optimizer_updates']==100000 and spec['watcher_sha256']==WATCHER_SHA and spec['supervisor_sha256']==CTRL_SUP_SHA
 assert spec['author_fixture_binding'] and spec['document_fixture_binding'] and spec['document_summary_sha256']
 for n,h in request['bound_sources'].items():assert sha(owned(ROOT/n))==h
 for n,h in spec['bound_sources'].items():assert sha(owned(ROOT/n))==h
 data=owned(ROOT/'step34_numeric_v1');out=owned(data/RUN);ctrl_out=owned(data/CONTROL)
 producer=read(out/'receipt.json');ctrl=read(ctrl_out/'receipt.json')
 intent=read(ctrl_out/'signal_intent.json');sent=read(ctrl_out/'signal_sent.json');before=read(ctrl_out/'producer_before_signal.json')
 cp=validate_signal_evidence(ctrl,request,intent,sent,before,producer)
 assert read(ctrl_out/'producer_final_snapshot.json')==producer
 trainerlog=owned(data/(RUN+'_supervision'));prefix='step4_train_only_'+RUN
 trainer_paths=dict(receipt=out/'receipt.json',exit=trainerlog/'exit.json',process=trainerlog/'process_exit.json',outer=ROOT/'receipts'/(prefix+'_outer_exit.json'))
 trainer={k:read(v) for k,v in trainer_paths.items()};th={k:sha(v) for k,v in trainer_paths.items()}
 verify_exit(trainer['process'],False);verify_exit(trainer['exit'],False)
 assert trainer['outer']['actual_returncode']!=0 and trainer['exit']['source_sha256']==SUP_SHA
 assert trainer['exit']['request_sha256']==REQUEST_SHA and trainer['exit']['receipt_sha256']==th['receipt']
 assert trainer['exit']['process_exit_sha256']==th['process'] and ctrl['original_exit_sha256']=={k:th[k] for k in ['exit','process','outer']}
 assert ctrl['original_final_receipt_sha256']==th['receipt']
 assert trainer['exit']['resource_sha256']==sha(owned(trainerlog/'resources.jsonl'))
 trainer_last=json.loads((trainerlog/'resources.jsonl').read_text().splitlines()[-1])
 assert trainer_last['phase']=='after_exit' and trainer_last['owned_group_processes']==trainer_last['owned_group_gpu_memory']==[]
 cprefix='ppo5000_control_'+CONTROL
 cpaths=dict(receipt=ctrl_out/'receipt.json',exit=ROOT/'receipts'/(cprefix+'_exit.json'),
  process=ROOT/'receipts'/(cprefix+'_process_exit.json'),outer=ROOT/'receipts'/(cprefix+'_outer_exit.json'))
 control={k:read(v) for k,v in cpaths.items()};ch={k:sha(v) for k,v in cpaths.items()}
 verify_exit(control['process'],True);verify_exit(control['exit'],True)
 assert control['outer']['actual_returncode']==0 and control['exit']['source_sha256']==CTRL_SUP_SHA
 assert control['exit']['spec_sha256']==CONTROL_SPEC_SHA and control['exit']['receipt_sha256']==ch['receipt']
 assert control['exit']['process_exit_sha256']==ch['process'] and control['exit']['model_lock_acquired'] is False
 cr=owned(ROOT/'logs'/(cprefix+'_resources.jsonl'))
 assert control['exit']['resource_sha256']==sha(cr)
 clast=json.loads(cr.read_text().splitlines()[-1]);assert clast['phase']=='after_exit' and clast['owned_group_processes']==clast['owned_group_gpu_memory']==[]
 # A following CPU supervisor will hold the common lock. Check the specific
 # stopped workers, never demand the global lock be free inside that new job.
 for process in pathlib.Path('/proc').iterdir():
  if not process.name.isdigit() or int(process.name)==os.getpid():continue
  try:
   if process.stat().st_uid!=1013:continue
   argv=(process/'cmdline').read_bytes().decode().split('\0')
   assert not (str(REQUEST) in argv and any(str(ROOT/'scripts'/n) in argv for n in ['run_step4_train_only_ppo_v4.py','supervise_step4_train_only_ppo_v4.py']))
   assert not (str(CONTROL_SPEC) in argv and any(str(ROOT/'scripts'/n) in argv for n in ['watch_ppo5000_endpoint.py','supervise_ppo5000_endpoint.py']))
  except (FileNotFoundError,ProcessLookupError):pass
 f=fingerprint(cp['path']);assert f['bytes']==cp['bytes'] and f['sha256']==cp['sha256']
 actual=dict(kind='ORIGINAL_TRAIN_PPO5000_INTERRUPTED_STATE_UNAUDITED',run=RUN,
  request_sha256=REQUEST_SHA,control_spec_sha256=CONTROL_SPEC_SHA,trainer_closure_sha256=th,
  controller_closure_sha256=ch,signal_intent_sha256=sha(ctrl_out/'signal_intent.json'),
  signal_sent_sha256=sha(ctrl_out/'signal_sent.json'),before_signal_sha256=sha(ctrl_out/'producer_before_signal.json'),
  input_library_binding=producer['input_library_binding'],checkpoint=cp,checkpoint_fingerprint=f,
  executor_sha256=ENTRY_SHA,runtime_sha256=RUNTIME_SHA,supervisor_sha256=SUP_SHA,
  original_exit_codes_retained=True,actual_checkpoint_steps_not_yet_audited=True)
 if binding is not None:assert binding==actual
 return dict(binding=actual,request=request,producer=producer,controller=ctrl)

def validate_fullstate_metadata(state,binding):
 assert state['schema']=='textop.stage4_train_only_fullstate.v1'
 assert type(state['completed_iterations']) is int and state['completed_iterations']==5000
 assert type(state['optimizer_updates']) is int and state['optimizer_updates']==100000
 assert state['input_binding']==binding['input_library_binding']
 assert state['physics_continuation']=='new episodes; simulator caches/in-flight episodes not serialized'
 assert state['algorithm']=='fresh one-stage PPO; no teacher/DAgger/distillation'
 assert set(state['rng'])=={'python','numpy','cpu','cuda_device1'}
 assert set(state['runner_clock'])=={'tot_time','tot_timesteps'}
 clock=state['runner_clock'];assert math.isfinite(clock['tot_time']) and clock['tot_time']>0
 assert type(clock['tot_timesteps']) is int and clock['tot_timesteps']==5000*24*8192
 assert math.isfinite(state['learning_rate']) and state['learning_rate']>0
 assert set(state['adaptive_sampling'])=={'failed_motion_count','success_motion_count','current_failed_motion_count',
  'current_success_motion_count','failed_total_count','success_total_count','raw_failed','raw_success'}
 assert set(state['obs_normalizer'])==set(state['critic_normalizer'])=={'_mean','_var','_std','count'}
 assert isinstance(state['model'],dict) and state['model']
 assert set(state['optimizer'])=={'state','param_groups'} and state['optimizer']['state']
 return dict(completed_iterations=5000,Adam_updates=100000,CUDA_RNG_restored=False,
  physics_trajectory_replay_claimed=False)

def accepted_export(binding=None):
 """Accept an actual bounded CPU export with separate real closure proof."""
 data=owned(ROOT/'step34_numeric_v1');run='user_step4_ppo5000_frozen_export_v1'
 out=owned(data/run);log=owned(data/(run+'_supervision'))
 prefix='ppo5000_frozen_'+run
 sp=owned(ROOT/'receipts'/(prefix+'_request.json'));s=read(sp)
 assert s['authorized'] is True and s['run']==run and s['role']=='export'
 assert s['training_binding'] and s['export_binding'] is None
 paths=dict(receipt=out/'receipt.json',exit=log/'exit.json',process=log/'process_exit.json',
  outer=ROOT/'receipts'/(prefix+'_outer_exit.json'))
 files={k:read(v) for k,v in paths.items()};hashes={k:sha(v) for k,v in paths.items()}
 for name in ['exit','process']:verify_exit(files[name],True)
 ex=files['exit'];rec=files['receipt']
 assert files['outer']['actual_returncode']==0 and ex['source_sha256']==s['supervisor_sha256']
 assert ex['request_sha256']==sha(sp) and ex['receipt_sha256']==hashes['receipt']
 assert ex['process_exit_sha256']==hashes['process'] and ex['resource_sha256']==sha(owned(log/'resources.jsonl'))
 last=json.loads((log/'resources.jsonl').read_text().splitlines()[-1])
 assert last['phase']=='after_exit' and last['owned_group_processes']==last['owned_group_gpu_memory']==[]
 assert rec['status']=='PPO5000_FROZEN_CPU_EXPORTED' and rec['output_writers_closed'] is True
 assert rec['source_sha256']==s['executor_sha256'] and rec['request_sha256']==sha(sp)
 assert rec['training_binding']==s['training_binding'] and rec['fullstate_exact_5000_100000'] is True
 assert rec['independent_actor_RMS_reload_exact'] is True and rec['synthetic_probe_only'] is True
 assert rec['optimizer_updates_performed']==rec['new_CUDA_contexts']==0 and rec['CUDA_RNG_restored'] is False
 artifact=rec['inference'];path=owned(artifact['path']);assert path==out/'actor_RMS.private.pth'
 fp=fingerprint(path);assert fp['bytes']==artifact['bytes'] and fp['sha256']==artifact['sha256'] and 0<fp['bytes']<=32*1024**2
 actual=dict(run=run,request_sha256=sha(sp),closure_sha256=hashes,inference=artifact,
  training_binding=s['training_binding'])
 if binding is not None:assert binding==actual
 return dict(binding=actual,receipt=rec,request=s)

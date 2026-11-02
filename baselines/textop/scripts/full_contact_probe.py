"""One real 8192-env PPO iteration on the complete available reference library.

Diagnostic copy of the frozen capacity probe, adding read-only numerical telemetry.
This is not a stability acceptance, a trained policy, or a physical filter.
The original references and sampling/recording class match the formal trainer.
It creates no checkpoint and changes no installed package or shared source.
"""
import argparse,hashlib,importlib.metadata,json,math,os,pathlib,shutil,time,traceback
from isaaclab.app import AppLauncher
from full_library_inputs import inspect,read,sha,ROOT

p=argparse.ArgumentParser()
p.add_argument('--spec',required=True);p.add_argument('--capacity-spec',required=True)
p.add_argument('--output',required=True);p.add_argument('--diagnostic-spec',required=True);p.add_argument('--case',required=True,choices=['baseline','no_self_collision','no_floor'])
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
receipt={'status':'STARTING','optimizer_updates':0,'new_checkpoints_written':0,
         'full_corpus_capacity_passed':False,'formal_filter_training_started':False,
         'full_generator_training_started':False,'physical_filter_executed':False,
         'dataset_accepted':False,'output_writers_closed':False}
app=None;env=None;runner=None;started=time.monotonic()

def save():
 receipt['elapsed_seconds']=time.monotonic()-started
 partial=out/'receipt.json.partial';partial.write_text(json.dumps(receipt,indent=2));partial.replace(out/'receipt.json')

try:
 diag=read(a.diagnostic_spec)
 assert sha(ROOT/'scripts/audit_full_contact_state.py')==diag['native_audit_sha256']
 for name,expected in diag['private_input_sha256'].items():
  assert sha(ROOT/diag['private_input_directory']/name)==expected
 assert sha(ROOT/'scripts/full_library_capacity_probe.py')==diag['base_probe_sha256']
 assert sha(ROOT/'scripts/numerical_trace.py')==diag['telemetry_sha256']
 assert sha(a.capacity_spec)==diag['capacity_spec_sha256']
 spec=read(a.spec);cap=read(a.capacity_spec);plan=read(ROOT/'configs/filter-training-plan.json')
 assert cap['launch_enabled'] is True,'Full-library diagnostic has not been activated'
 assert a.headless and os.getuid()==1013 and a.device==cap['device']=='cuda:1'
 assert set(os.sched_getaffinity(0))<={0,1,2,3} and cap['cpu_threads']==4
 assert hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]==cap['warm_environment_canonical_hash']
 assert all(importlib.metadata.version(k)==v for k,v in spec['required_versions'].items())
 assert sha(spec['robot_urdf'])==spec['robot_urdf_sha256']
 assert out.stat().st_uid==1013 and shutil.disk_usage(out).free>cap['minimum_free_data_disk_gib']*1024**3
 assert [x['run'] for x in cap['cohorts']]==plan['input_reference_exports']
 assert cap['num_envs']==plan['num_envs']==8192 and cap['expected_original_sources']==plan['available_public_source_target']==8278
 proof=cap['persistent_backup_acceptance'];backup=pathlib.Path(proof['receipt'])
 assert sha(backup)==proof['sha256'];b=read(backup)
 assert b['status']=='PREPROCESSING_ALL_AVAILABLE_SOURCES_BACKED_UP_VERIFIED_FILTER_PENDING'
 assert b['preprocessing_complete'] is True and b['verified_sources']==8278 and b['output_writers_closed'] is True
 assert b['physical_filter_executed'] is False and b['dataset_accepted'] is False
 rows,inputs=inspect(cap,read(ROOT/'configs/paper-step2.json')['data_scope'],require_complete=True)
 receipt.update(source_sha256=sha(__file__),capacity_spec_sha256=sha(a.capacity_spec),
                input_verifier_sha256=sha(ROOT/'scripts/full_library_inputs.py'),
                recording_runtime_sha256=sha(ROOT/'scripts/filter_training_runtime.py'),
                paper_config_sha256=sha(ROOT/'scripts/paper_filter_config.py'),
                backup_audit_sha256=proof['sha256'],inputs=inputs,num_envs=8192,
                reference_sources=len(rows),reference_frames=sum(x['frames'] for x in rows),
                cpu_affinity=sorted(os.sched_getaffinity(0)),paper_data_scope_exact=False)
 save();app=AppLauncher(a).app
 import torch,gymnasium as gym
 from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
 from rsl_rl.runners import OnPolicyRunner
 from paper_filter_config import make,paper_settings
 from filter_training_runtime import RecordedMotionCommand,fingerprint
 torch.set_num_threads(4);torch.manual_seed(cap['seed']);torch.cuda.manual_seed_all(cap['seed'])
 torch.cuda.set_per_process_memory_fraction(cap['torch_memory_fraction'],torch.device(a.device))
 # Keep the same Torch defaults as train_filter_policy.py; do not introduce a
 # separate TF32/cuDNN execution profile just to make the probe fit or pass.
 free,total=torch.cuda.mem_get_info(a.device);assert free>cap['minimum_free_gpu_gib']*1024**3
 assert fingerprint(rows)==inputs['reference_order_fingerprint']
 receipt.update(status='BUILDING_FULL_LIBRARY_8192_ENVIRONMENTS',device_free_before_environment=free,
                device_total_bytes=total,torch_allocator_fraction_configured=cap['torch_memory_fraction'],
                torch_threads=torch.get_num_threads(),torch_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
 save()
 cfg,agent=make(spec,[x['file'] for x in rows],out/'usd',phase='relaxed',num_envs=8192)
 cfg.commands.motion.class_type=RecordedMotionCommand
 if a.case=='no_self_collision':cfg.scene.robot.spawn.articulation_props.enabled_self_collisions=False
 if a.case=='no_floor':cfg.scene.terrain=None
 receipt['contact_case']=a.case
 assert cfg.terminations.anchor_pos is None and cfg.terminations.ee_body_pos is None
 assert agent.num_steps_per_env==cap['rollout_steps']==24
 assert agent.policy.actor_hidden_dims==agent.policy.critic_hidden_dims==cap['actor_critic_hidden_dims']==[2048,1024,512]
 env=RslRlVecEnvWrapper(gym.make('Tracking-Flat-G1-PrivPrivObs-v0',cfg=cfg))
 obs,info=env.get_observations();motion=env.unwrapped.command_manager.get_term('motion')
 assert obs.shape==info['observations']['critic'].shape==(8192,557)
 assert motion.num_motion==8278 and int((motion.raw_failed+motion.raw_success).sum())==0
 receipt.update(status='ENVIRONMENT_READY_BUILDING_PPO',device_free_after_environment=torch.cuda.mem_get_info(a.device)[0]);save()
 runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(out/'training_logs'),device=a.device)
 from audit_full_contact_state import audit
 receipt['native_state_audit']=audit(env,ROOT/diag['private_input_directory'],out)
 from numerical_trace import install
 finish_trace=install(env,out,rows,diag)
 before=[x.detach().cpu().clone() for x in runner.alg.policy.parameters()]
 original_step=runner.alg.optimizer.step;original_log=runner.log
 def step(*args,**kwargs):
  assert all(torch.isfinite(x).all() and (x.grad is None or torch.isfinite(x.grad).all()) for x in runner.alg.policy.parameters())
  result=original_step(*args,**kwargs);receipt['optimizer_updates']+=1;return result
 def log(values,*args,**kwargs):
  original_log(values,*args,**kwargs)
  losses={k:float(v) for k,v in values['loss_dict'].items()}
  assert all(math.isfinite(x) for x in losses.values());receipt['losses']=losses;save()
 def suppressed_save(*args,**kwargs):
  receipt['automatic_checkpoint_calls_suppressed']=receipt.get('automatic_checkpoint_calls_suppressed',0)+1
 runner.alg.optimizer.step=step;runner.log=log;runner.save=suppressed_save
 # Preserve the seeded collection path; no optimizer update is needed to diagnose physics.
 runner.alg.update=lambda: {}
 receipt['status']='BOUNDED_FULL_SCENE_CONTACT_DIAGNOSIS_ZERO_OPTIMIZER_UPDATES';save()
 runner.learn(num_learning_iterations=cap['learning_iterations'],init_at_random_ep_len=False)
 receipt['numerical_trace']=finish_trace()
 assert receipt['numerical_trace']['reward_samples']==24
 assert receipt['numerical_trace']['physics_samples']==24*cfg.decimation
 assert cap['learning_iterations']==1 and receipt['optimizer_updates']==diag['optimizer_updates']==0
 assert all(torch.isfinite(x).all() for x in runner.alg.policy.parameters())
 change=max(float((x-y.detach().cpu()).abs().max()) for x,y in zip(before,runner.alg.policy.parameters()));assert change==0
 runner.writer.flush();runner.writer.close();runner.writer=None
 assert not list(out.rglob('*.pt')),'Capacity-only run must not create another checkpoint'
 receipt.update(status='TEXTOP_FULL_CONTACT_DIAGNOSTIC_COMPLETED',full_corpus_capacity_passed=False,
                completed_learning_iterations=0,completed_collection_iterations=1,environment_transitions=8192*24,
                policy_observation_dim=557,critic_observation_dim=557,parameter_max_change=change,
                initial_resets_excluded_from_raw_counts=True,
                raw_completed_attempts=int((motion.raw_failed+motion.raw_success).sum()),
                peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
                peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(a.device),
                output_writers_closed=True,paper_settings=paper_settings(),
                limitation='All8278 available original references, one real PPO iteration only; no trained-policy quality or motion acceptance claim')
 save();print(receipt['status'],flush=True)
except BaseException as error:
 receipt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());save();raise
finally:
 if runner is not None and runner.writer is not None:runner.writer.flush();runner.writer.close();runner.writer=None
 if app is not None:
  from bounded_headless_close import close
  close(app,receipt,save,out,simulation=env.unwrapped.sim if env is not None else None,
        successful_statuses=('TEXTOP_FULL_CONTACT_DIAGNOSTIC_COMPLETED',))

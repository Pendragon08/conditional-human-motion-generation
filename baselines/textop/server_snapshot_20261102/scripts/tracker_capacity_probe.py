"""8192 environments, completed643-source library and one real PPO iteration.

Capacity diagnostic only: this does not filter motions, accept a dataset or
replace40000 learning iterations. No package/shared-source changes.
"""
import argparse,hashlib,importlib.metadata,json,os,pathlib,shutil,time,traceback
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--capacity-spec',required=True);p.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
receipt={'status':'STARTING','optimizer_updates':0,'dataset_accepted':False,'physical_filter_executed':False,
         'formal_filter_training_started':False,'full_generator_training_started':False,'full_corpus_capacity_passed':False,
         'output_writers_closed':False};app=None;env=None;runner=None;started=time.monotonic()
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def save():
 receipt['elapsed_seconds']=time.monotonic()-started
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2))
try:
 spec=json.loads(pathlib.Path(a.spec).read_text());cap=json.loads(pathlib.Path(a.capacity_spec).read_text())
 assert hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]==cap['warm_environment_canonical_hash']
 assert a.headless and a.device==cap['device']=='cuda:1' and os.getuid()==1013
 assert len(os.sched_getaffinity(0))<=4
 assert shutil.disk_usage(out).free>cap['minimum_free_data_disk_gib']*1024**3
 assert all(importlib.metadata.version(k)==v for k,v in spec['required_versions'].items())
 assert sha(spec['robot_urdf'])==spec['robot_urdf_sha256']
 references=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references')/cap['reference_run']
 meta=json.loads((references/'receipt.json').read_text());assert meta['status']=='TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED'
 assert meta['exported_source_sequences']==cap['expected_source_sequences'] and meta['exported_trajectories']==cap['expected_reference_trajectories']
 assert meta['exported_frames']==cap['expected_reference_frames'] and sha(references/'manifest.private.jsonl')==meta['private_manifest_sha256']
 rows=[json.loads(x) for x in (references/'manifest.private.jsonl').read_text().splitlines()]
 assert len(rows)==1286 and all(sha(x['file'])==x['sha256'] for x in rows)
 receipt.update(source_sha256=sha(__file__),capacity_spec_sha256=sha(a.capacity_spec),cpu_affinity=sorted(os.sched_getaffinity(0)),
                num_envs=cap['num_envs'],reference_source_sequences=643,reference_trajectories=len(rows),reference_frames=946700,
                private_reference_manifest_sha256=meta['private_manifest_sha256'])
 save();app=AppLauncher(a).app
 import torch,gymnasium as gym
 from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
 from rsl_rl.runners import OnPolicyRunner
 from paper_filter_config import make,paper_settings
 torch.set_num_threads(4);torch.manual_seed(cap['seed']);torch.cuda.manual_seed_all(cap['seed'])
 torch.cuda.set_per_process_memory_fraction(cap['torch_memory_fraction'],torch.device(a.device))
 torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False
 free,total=torch.cuda.mem_get_info(a.device);assert free>cap['minimum_free_gpu_gib']*1024**3
 receipt.update(status='BUILDING_8192_ENVIRONMENTS',torch_threads=torch.get_num_threads(),torch_allocator_fraction_configured=.65,
                gpu_name=torch.cuda.get_device_name(a.device),device_free_before_environment=free,device_total_bytes=total)
 save()
 cfg,agent=make(spec,[x['file'] for x in rows],out/'usd',phase='relaxed',num_envs=cap['num_envs'])
 assert cfg.terminations.anchor_pos is None and cfg.terminations.ee_body_pos is None
 assert agent.num_steps_per_env==24 and agent.policy.actor_hidden_dims==agent.policy.critic_hidden_dims==[2048,1024,512]
 env=RslRlVecEnvWrapper(gym.make('Tracking-Flat-G1-PrivPrivObs-v0',cfg=cfg));obs,info=env.get_observations()
 assert obs.shape==(8192,557) and info['observations']['critic'].shape==(8192,557)
 receipt.update(status='ENVIRONMENT_READY_BUILDING_PPO',device_free_after_environment=torch.cuda.mem_get_info(a.device)[0]);save()
 runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(out/'training_logs'),device=a.device)
 before=[p.detach().clone() for p in runner.alg.policy.parameters()];original=runner.alg.optimizer.step
 def step(*args,**kwargs):
  assert all(torch.isfinite(x).all() and (x.grad is None or torch.isfinite(x.grad).all()) for x in runner.alg.policy.parameters())
  result=original(*args,**kwargs);receipt['optimizer_updates']+=1;return result
 runner.alg.optimizer.step=step;receipt['status']='BOUNDED_ONE_ITERATION_PPO';save()
 runner.learn(num_learning_iterations=1,init_at_random_ep_len=False)
 assert receipt['optimizer_updates']==20
 change=max(float((x-y).abs().max()) for x,y in zip(before,runner.alg.policy.parameters()));assert change>0
 runner.current_learning_iteration=1;checkpoint=out/'capacity_completed_iteration_1.pt'
 runner.save(str(checkpoint),infos={'diagnostic_only':True,'completed_iterations':1,'actual_optimizer_updates':20})
 runner.writer.flush();runner.writer.close();runner.writer=None
 receipt.update(status='TEXTOP_8192_CAPACITY_PROBE_PASSED',completed_learning_iterations=1,
                environment_transitions=8192*24,policy_observation_dim=557,critic_observation_dim=557,
                parameter_max_change=change,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
                peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(a.device),
                checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256=sha(checkpoint),
                output_writers_closed=True,paper_settings=paper_settings(),
                limitation=cap['limitation'],config_source_sha256=sha(pathlib.Path(__file__).with_name('paper_filter_config.py')))
 save();print('TEXTOP_8192_CAPACITY_PROBE_PASSED',flush=True)
except BaseException as error:
 receipt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());save();raise
finally:
 if runner is not None and runner.writer is not None:runner.writer.flush();runner.writer.close()
 if app is not None:
  from bounded_headless_close import close
  close(app,receipt,save,out,simulation=env.unwrapped.sim if env is not None else None,
        successful_statuses=('TEXTOP_8192_CAPACITY_PROBE_PASSED',))

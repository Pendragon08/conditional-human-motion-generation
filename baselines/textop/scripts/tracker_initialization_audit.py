"""Diagnose reference/reset geometry and raw-PD contacts; no acceptance or PPO.

Own deterministic diagnostic. Pinned upstream and formal paper settings are
unchanged. Per-environment values stay private; only aggregate receipts publish.
"""
import argparse,hashlib,json,os,pathlib,time,traceback
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--audit-spec',required=True);p.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
receipt={'status':'STARTING','optimizer_updates':0,'dataset_accepted':False,'physical_filter_executed':False,
         'formal_filter_training_started':False,'full_generator_training_started':False,'output_writers_closed':False}
app=None;env=None;started=time.monotonic()
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def save():
 receipt['elapsed_seconds']=time.monotonic()-started
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2))
try:
 spec=json.loads(pathlib.Path(a.spec).read_text());audit=json.loads(pathlib.Path(a.audit_spec).read_text())
 assert hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]==audit['warm_environment_canonical_hash']
 assert os.getuid()==1013 and a.headless and a.device==audit['device'] and len(os.sched_getaffinity(0))<=4
 refs=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references')/audit['reference_run']
 meta=json.loads((refs/'receipt.json').read_text());rows=[json.loads(x) for x in (refs/'manifest.private.jsonl').read_text().splitlines()]
 assert meta['status']=='TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED' and len(rows)==audit['expected_trajectories']
 assert sha(refs/'manifest.private.jsonl')==meta['private_manifest_sha256'] and all(sha(x['file'])==x['sha256'] for x in rows)
 receipt.update(source_sha256=sha(__file__),audit_spec_sha256=sha(a.audit_spec),cpu_affinity=sorted(os.sched_getaffinity(0)),
                reference_trajectories=len(rows),private_manifest_sha256=meta['private_manifest_sha256'],diagnostic_changes=audit['changes_for_diagnosis_only'])
 save();app=AppLauncher(a).app
 import numpy as np,torch,gymnasium as gym
 from paper_filter_config import make
 from textop_tracker.tasks.tracking.mdp.commands_multi import MotionCommand
 torch.set_num_threads(4);torch.manual_seed(audit['seed']);torch.cuda.manual_seed_all(audit['seed'])
 torch.cuda.set_per_process_memory_fraction(audit['torch_memory_fraction'],torch.device(a.device))
 assert torch.cuda.mem_get_info(a.device)[0]>20*1024**3
 class AssignedCommand(MotionCommand):
  def _uniform_sampling(self,env_ids):return env_ids%self.num_motion
 cfg,_=make(spec,[x['file'] for x in rows],out/'usd',phase='relaxed',num_envs=len(rows))
 cfg.commands.motion.class_type=AssignedCommand;cfg.commands.motion.enable_adaptive_sampling=False
 cfg.commands.motion.start_from_zero_step=True;cfg.commands.motion.pose_range={};cfg.commands.motion.velocity_range={}
 cfg.commands.motion.joint_position_range=(0.,0.);cfg.commands.motion.random_static_prob=-1.
 env=gym.make('Tracking-Flat-G1-PrivPrivObs-v0',cfg=cfg);raw=env.unwrapped
 env.reset();motion=raw.command_manager.get_term('motion');robot=raw.scene['robot']
 raw.sim.forward();raw.scene.update(raw.physics_dt)
 def distribution(value):
  array=value.detach().float().cpu().numpy().reshape(-1);assert np.isfinite(array).all()
  return {'count':int(len(array)),'percentiles_0_50_95_99_100':np.percentile(array,[0,50,95,99,100]).tolist()}
 def initial_state():
  error=(motion.body_pos_w-motion.robot_body_pos_w).norm(dim=-1)
  jerror=(motion.joint_pos-motion.robot_joint_pos).abs()
  return {'root_position_error_m':distribution((motion.body_pos_w[:,0]-motion.robot_anchor_pos_w).norm(dim=-1)),
          'tracked_body_error_m':distribution(error),'joint_clipping_difference_rad':distribution(jerror),
          'envs_with_joint_clipping':int((jerror.max(dim=1).values>1e-5).sum()),
          'initial_root_speed_m_s':distribution(robot.data.root_lin_vel_w.norm(dim=-1)),
          'initial_joint_speed_rad_s':distribution(robot.data.joint_vel.abs()),
          'minimum_tracked_body_origin_z_m':distribution(motion.body_pos_w[:,:,2].min(dim=1).values)}
 receipt['initial_frame']=initial_state();save()
 assert receipt['initial_frame']['root_position_error_m']['percentiles_0_50_95_99_100'][-1]<1e-4
 action_term=raw.action_manager.get_term('joint_pos')
 receipt['action_scale_range']=distribution(torch.as_tensor(action_term._scale,device=a.device))
 steps=[]
 with torch.inference_mode():
  for index in range(audit['pd_control_steps']):
   # Exactly undo the installed JointPositionAction scale/default offset.
   actions=(motion.joint_pos-action_term._offset)/action_term._scale
   _,reward,terminated,truncated,_=env.step(actions)
   forces=raw.scene['contact_forces'].data.net_forces_w.norm(dim=-1)
   steps.append({'step':index+1,'contact_force_n':distribution(forces),
                 'joint_speed_rad_s':distribution(robot.data.joint_vel.abs()),
                 'body_position_error_m':distribution((motion.body_pos_w-motion.robot_body_pos_w).norm(dim=-1)),
                 'termination_events':int(terminated.sum()),'timeout_events':int(truncated.sum()),
                 'reward':distribution(reward)})
 receipt.update(status='TEXTOP_INITIALIZATION_AUDIT_FINISHED',pd_steps=steps,output_writers_closed=True,
                peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
                limitation='Untrained reference-PD diagnostic, not learned tracking or a physical acceptance decision')
 save();print('TEXTOP_INITIALIZATION_AUDIT_FINISHED',flush=True)
except BaseException as error:
 receipt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());save();raise
finally:
 if app is not None:
  from bounded_headless_close import close
  close(app,receipt,save,out,simulation=env.unwrapped.sim if env is not None else None,
        successful_statuses=('TEXTOP_INITIALIZATION_AUDIT_FINISHED',))

"""Two separate SDK stages: tracker-URDF FK, then actual generated-ref tracking."""
import argparse,importlib.metadata,json,os,pathlib,shutil,sys,time,traceback
from isaaclab.app import AppLauncher
from generated_pipeline_protocol import gate,sha
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['export','replay'],required=True);p.add_argument('--run',required=True);AppLauncher.add_app_launcher_args(p);a=p.parse_args()
sp=ROOT/'configs/quick-generated-sdk-spec.json';spec=json.loads(sp.read_text());item=gate(ROOT,spec)
assert a.headless and a.device=='cuda:1' and os.getuid()==1013 and set(os.sched_getaffinity(0))=={0,1,2,3} and a.run.isidentifier()
assert sha(__file__)==spec['executor_sha256'];warm=json.loads((ROOT/'configs/tracker-env-spec.json').read_text());assert sha(warm['robot_urdf'])==warm['robot_urdf_sha256']
assert sys.version.split()[0]==warm['python_exact'] and {k:importlib.metadata.version(k) for k in warm['required_versions']}==warm['required_versions']
out=ROOT/'generated_execution_private'/a.run/a.stage;out.mkdir(parents=True,mode=0o700,exist_ok=False);started=time.monotonic()
r={'status':'STARTING_GENERATED_'+a.stage.upper(),'stage':a.stage,'quality_label':'QUICK_REPRODUCTION_NOT_PAPER_COMPLETE','source_kind':'trained_generated_reference',
 'optimizer_updates':0,'source_sha256':sha(__file__),'spec_sha256':sha(sp),'input_generated_npz_sha256':item['sha256'],'output_writers_closed':False,
 'whole_library_filtered':False,'real_time_interactive_closed_loop_verified':False,'final_reference_endpoint_comparison_performed':False}
def save():
 r['elapsed_seconds']=time.monotonic()-started;tmp=out/'receipt.partial';tmp.write_text(json.dumps(r,indent=2,allow_nan=False));tmp.replace(out/'receipt.json')
save();app=None;sim=None;gym_env=None;trace=None
try:
 import numpy as np
 with np.load(item['file'],allow_pickle=False) as z:q=np.array(z['qpos_29'],dtype=np.float32)
 metadata=json.loads((pathlib.Path(item['file']).parent/'generation.private.json').read_text());assert q.shape==(258,36) and np.isfinite(q).all() and len(metadata['joint_names'])==29
 assert shutil.disk_usage(ROOT).free>2*1024**3+64*1024**2
 app=AppLauncher(a).app
 import torch
 torch.set_num_threads(4);torch.manual_seed(20261026);torch.cuda.manual_seed_all(20261026);torch.cuda.set_per_process_memory_fraction(.65,torch.device(a.device))
 assert torch.cuda.mem_get_info(a.device)[0]>20*1024**3
 if a.stage=='export':
  import isaaclab.sim as sim_utils
  from isaaclab.assets import ArticulationCfg
  from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
  from isaaclab.utils import configclass
  from textop_tracker.robots.g1 import G1_CYLINDER_CFG
  robot_cfg=G1_CYLINDER_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot');robot_cfg.spawn.asset_path=warm['robot_urdf'];robot_cfg.spawn.usd_dir=str(out/'usd');robot_cfg.spawn.usd_file_name='g1.usd'
  @configclass
  class Scene(InteractiveSceneCfg):robot:ArticulationCfg=robot_cfg
  sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device=a.device));scene=InteractiveScene(Scene(num_envs=128,env_spacing=2.5));sim.reset();robot=scene['robot']
  assert len(robot.joint_names)==29 and len(robot.body_names)==30
  order=[metadata['joint_names'].index(n) for n in robot.joint_names];jp=q[:,7:][:,order];positions=[];quaternions=[]
  with torch.inference_mode():
   for start in range(0,len(q),128):
    n=min(128,len(q)-start);idx=np.minimum(np.arange(128)+start,len(q)-1)
    root=robot.data.default_root_state.clone();root[:,:3]=torch.as_tensor(q[idx,:3],device=a.device)+scene.env_origins;root[:,3:7]=torch.as_tensor(q[idx,3:7],device=a.device);root[:,7:]=0
    joints=torch.as_tensor(jp[idx],device=a.device);robot.write_root_state_to_sim(root);robot.write_joint_state_to_sim(joints,torch.zeros_like(joints));sim.forward();scene.update(.005)
    assert torch.allclose(robot.data.joint_pos,joints,atol=1e-5)
    positions.append((robot.data.body_pos_w-scene.env_origins[:,None,:])[:n].cpu().numpy());quaternions.append(robot.data.body_quat_w[:n].cpu().numpy())
  bp=np.concatenate(positions);bq=np.concatenate(quaternions);assert bp.shape==(258,30,3) and bq.shape==(258,30,4) and np.isfinite(bp).all() and np.isfinite(bq).all()
  assert np.max(np.abs(np.linalg.norm(bq,axis=-1)-1))<1e-4
  prev=bq[:-1];nxt=bq[1:];w=np.sum(prev*nxt,axis=-1);v=prev[...,:1]*nxt[...,1:]-nxt[...,:1]*prev[...,1:]+np.cross(prev[...,1:],nxt[...,1:]);sign=np.where(w<0,-1.,1.);w=w*sign;v=v*sign[...,None];norm=np.linalg.norm(v,axis=-1)
  adjacent=v*(2*np.arctan2(norm,w)/np.maximum(norm,1e-12))[...,None]/.02;av=np.concatenate([adjacent[:1],(adjacent[:-1]+adjacent[1:])*.5,adjacent[-1:]],axis=0)
  dest=out/'reference.private.npz';np.savez_compressed(dest,fps=np.array([50]),joint_pos=jp,joint_vel=np.gradient(jp,.02,axis=0),body_pos_w=bp,body_quat_w=bq,body_lin_vel_w=np.gradient(bp,.02,axis=0),body_ang_vel_w=av)
  r.update(status='GENERATED_REFERENCE_TRACKER_URDF_FK_EXPORTED',frames=258,joint_names=robot.joint_names,body_names=robot.body_names,urdf_sha256=warm['robot_urdf_sha256'],
   reference={'file':str(dest),'sha256':sha(dest),'bytes':dest.stat().st_size},explicit_physics_step_calls=0,physical_tracking_attempted=False,output_writers_closed=True)
 else:
  from quick_pilot_protocol import load_teacher,authorize_teacher
  from paper_filter_config import make
  from validated_contact_profile import apply as apply_contact_profile
  from apply_tracking_filter_v3 import contact_signature
  from strict_tracking_coverage import StrictCoverage
  from quick_pilot_trace import ExecutionTrace
  from textop_tracker.tasks.tracking.mdp.commands_multi import MotionCommand
  from isaaclab.managers import TerminationTermCfg
  from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
  from isaaclab.utils.math import yaw_quat,quat_mul,quat_inv,quat_apply
  from rsl_rl.runners import OnPolicyRunner
  import gymnasium as gym
  export=out.parent/'export';er=json.loads((export/'receipt.json').read_text());ex=json.loads((ROOT/'receipts'/('generated_pipeline_'+a.run+'_export_exit.json')).read_text())
  assert ex['supervision_passed'] and ex['child_exit_code']==0 and ex['evaluation_receipt_sha256']==sha(export/'receipt.json') and er['status']=='GENERATED_REFERENCE_TRACKER_URDF_FK_EXPORTED'
  ref=pathlib.Path(er['reference']['file']);assert sha(ref)==er['reference']['sha256'] and er['input_generated_npz_sha256']==item['sha256']
  quick=json.loads((ROOT/'configs/quick-pilot-spec.json').read_text());teacher=load_teacher(ROOT,quick);checkpoint=authorize_teacher(quick,teacher);assert sha(checkpoint['path'])==checkpoint['sha256']
  saved=torch.load(checkpoint['path'],map_location='cpu',weights_only=False);assert saved['infos']['completed_learning_iterations']==5000 and saved['infos']['actual_optimizer_updates']==100000;del saved
  def motion_complete(env,command_name='motion'):
   m=env.command_manager.get_term(command_name);return m.time_steps>=(m.motion_length-2).clamp_min(0)
  cfg,agent=make(warm,[str(ref)],out/'usd',phase='strict',num_envs=1);plan=json.loads((ROOT/'configs/filter-training-plan.json').read_text())
  r['contact_profile']=apply_contact_profile(cfg,out/'usd',ROOT,plan['contact_profile']);assert contact_signature(r['contact_profile'])==contact_signature(teacher['training']['contact_profile'])
  assert cfg.terminations.anchor_pos.params['threshold']==.25 and cfg.terminations.ee_body_pos.params['threshold']==.5 and cfg.terminations.anchor_ori is not None and cfg.decimation==4
  cfg.commands.motion.class_type=MotionCommand;cfg.commands.motion.enable_adaptive_sampling=False;cfg.commands.motion.start_from_zero_step=True
  cfg.commands.motion.pose_range={};cfg.commands.motion.velocity_range={};cfg.commands.motion.joint_position_range=(0.,0.);cfg.commands.motion.random_static_prob=-1.
  cfg.episode_length_s=258/50+.04;cfg.terminations.time_out=None;cfg.terminations.motion_complete=TerminationTermCfg(func=motion_complete,time_out=True)
  gym_env=gym.make('Tracking-Flat-G1-PrivPrivObs-v0',cfg=cfg);wrapped=RslRlVecEnvWrapper(gym_env,clip_actions=None);raw=gym_env.unwrapped;sim=raw.sim;motion=raw.command_manager.get_term('motion')
  assert wrapped.get_observations()[0].shape==(1,557) and motion.motion.file_lengths.tolist()==[258] and len(motion.cfg.body_names)==14
  assert motion.robot.joint_names==er['joint_names'] and motion.robot.body_names==er['body_names']
  runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=a.device);runner.load(checkpoint['path'],load_optimizer=False);policy=runner.get_inference_policy(device=a.device)
  def no_optimizer(*args,**kwargs):raise AssertionError('Generated replay must never optimize')
  runner.alg.optimizer.step=no_optimizer
  def align_reset():
   before=motion.time_steps.clone();n=len(motion.cfg.body_names);ap=motion.anchor_pos_w[:,None,:].repeat(1,n,1);aq=motion.anchor_quat_w[:,None,:].repeat(1,n,1)
   rp=motion.robot_anchor_pos_w[:,None,:].repeat(1,n,1).clone();rq=motion.robot_anchor_quat_w[:,None,:].repeat(1,n,1);rp[...,2]=ap[...,2];dq=yaw_quat(quat_mul(rq,quat_inv(aq)))
   motion.body_pos_relative_w=rp+quat_apply(dq,motion.body_pos_w-ap);motion.body_quat_relative_w=quat_mul(dq,motion.body_quat_w);assert torch.equal(before,motion.time_steps)
  def snapshot(actions):
   values={'reference_frame':motion.time_steps[:,None],'root_pos':motion.robot.data.root_pos_w-raw.scene.env_origins,'root_quat_wxyz':motion.robot.data.root_quat_w,
    'joint_pos':motion.robot_joint_pos,'reference_joint_pos':motion.joint_pos,'actions':actions,'body_pos':motion.robot_body_pos_w-raw.scene.env_origins[:,None,:],
    'reference_body_pos':motion.body_pos_relative_w-raw.scene.env_origins[:,None,:]}
   return {k:v.detach().cpu().numpy()[0].copy() for k,v in values.items()}
  captured={};capture={'active':False,'samples':0};original_compute=raw.termination_manager.compute
  def before_autoreset(*args,**kwargs):
   result=original_compute(*args,**kwargs)
   if capture['active']:
    captured['terms']={name:bool(raw.termination_manager.get_term(name)[0]) for name in raw.termination_manager.active_terms};captured['poses']=snapshot(capture['actions']);capture['samples']+=1
   return result
  raw.termination_manager.compute=before_autoreset
  budget={'used':0,'maximum':64*1024**2};trace=ExecutionTrace(out/'execution_private/trial_0',item['sha256'],258,motion.robot.joint_names,budget=budget);coverage=StrictCoverage(258,0)
  gym_env.reset();align_reset();observations=raw.observation_manager.compute();trace.append(snapshot(torch.zeros((1,29),device=a.device)))
  r.update(status='RUNNING_GENERATED_REFERENCE_PHYSICS',teacher_checkpoint_sha256=checkpoint['sha256'],teacher_completed_iterations=5000,reference_sha256=er['reference']['sha256'],num_envs=1);save()
  with torch.inference_mode():
   for step in range(257):
    actions=policy(observations['policy']);assert torch.isfinite(actions).all();capture.update(actions=actions,active=True);before=capture['samples']
    observations,rewards,terminated,truncated,extras=gym_env.step(actions);capture['active']=False;assert capture['samples']==before+1
    trace.append(captured['poses']);assert torch.isfinite(observations['policy']).all() and torch.isfinite(rewards).all()
    coverage.observe(terminated=bool(terminated[0]),motion_end=bool(truncated[0]));r.update(actual_vectorized_control_calls=step+1,actual_controlled_physics_substeps=(step+1)*4);save()
    if coverage.end_reason!='running':break
  trial=coverage.record();index=trace.finish(trial);ip=out/'executed_traces.private.json';ip.write_text(json.dumps([index],indent=2))
  r.update(status='GENERATED_REFERENCE_PHYSICS_TRIAL_FINISHED',trial=trial,output_writers_closed=True,terminal_execution_state_captured_before_autoreset=True,
   private_trace_index_sha256=sha(ip),private_trace_bytes=budget['used'],termination_terms=captured['terms'],own_tracking_metrics=trace.meta['metrics'],
   text_switch_reference_frame=130,text_switch_physically_consumed=trial['control_steps']-1>=130,
   limitation='Offline generated-reference execution, quick5k teacher/42-source VAE2k LDM3k, not real-time interactive finalWBC; final reference endpoint unchecked')
 r['peak_torch_allocated_bytes']=torch.cuda.max_memory_allocated(a.device);save();print(json.dumps(r),flush=True)
except BaseException as error:
 if trace is not None and not trace.stream.closed:trace.close_incomplete()
 r.update(status='FAILED_GENERATED_PIPELINE_SDK',error=repr(error),traceback=traceback.format_exc());save();raise
finally:
 if app is not None:
  from bounded_headless_close import close
  close(app,r,save,out,simulation=sim,successful_statuses=('GENERATED_REFERENCE_TRACKER_URDF_FK_EXPORTED','GENERATED_REFERENCE_PHYSICS_TRIAL_FINISHED'))

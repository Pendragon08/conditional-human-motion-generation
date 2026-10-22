"""First-episode frozen PhyGile evaluation in actual Unitree Mjlab physics."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,sys,time,traceback
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',WANDB_MODE='disabled')
p=argparse.ArgumentParser();p.add_argument('--unitree-checkout',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--clip',type=int,default=0);p.add_argument('--seed',type=int,default=2026)
p.add_argument('--checkpoint',type=Path);p.add_argument('--contact-profile',choices=['official','matched_sliding_friction'],default='official')
p.add_argument('--max-steps',type=int,default=0);p.add_argument('--device',default='cpu');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(a.unitree_checkout.resolve()))
receipt={'status':'STARTING','evaluation_type':'simulation_only','optimizer_steps':0,'seed':a.seed,'device':a.device,'clip':a.clip,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unitree_rl_mjlab_commit':'1425b15f73bd4095f0df53709d7c389c3eb9e790','paper_benchmark_reproduced':False}
def save():
 temp=a.output/'receipt.partial';temp.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');temp.replace(a.output/'receipt.json')
save();env=None
try:
 import numpy as np,torch,mujoco,warp as wp
 from mjlab_task import make_config,convert_motion,PhyGileMjlabEnv,initialize_relative
 from frozen_policy import FrozenPhyGile
 torch.set_num_threads(4);torch.manual_seed(a.seed);wp.init()
 c=json.loads((a.assets/'contract.json').read_text(encoding='utf-8'));source=a.assets/f'test_{a.clip:03d}.npz'
 with np.load(source,allow_pickle=False) as data:nframes=len(data['joint_pos'])
 cfg=make_config(c,a.output/'converted_motion.npz',nframes,a.seed,a.contact_profile)
 standalone=cfg.scene.entities['robot'].build()
 mapping=convert_motion(source,a.output/'converted_motion.npz',standalone.joint_names,standalone.body_names)
 checkpoint=a.checkpoint or a.assets/'parent_checkpoint.pt'
 receipt.update(mapping=mapping,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),motion_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),motion_frames=nframes,contact_profile=a.contact_profile)
 save()
 env=PhyGileMjlabEnv(cfg,c,device=a.device,render_mode=None)
 policy=FrozenPhyGile(checkpoint,c,a.assets/'encoder_actor_critic.py',a.device)
 from source_provenance import capture as capture_sources,verify as verify_sources
 import mjlab
 receipt['source_provenance']=capture_sources(a.output/'executed_sources.json',[Path(__file__).resolve().parents[1],a.unitree_checkout,Path(mjlab.__file__).parent],[__file__,a.assets/'contract.json',a.assets/'encoder_actor_critic.py',checkpoint,source])
 save()
 robot=env.scene['robot'];mt=env.command_manager.get_term('motion')
 obs,_=env.reset(seed=a.seed);initialize_relative(mt)
 obs=env.observation_manager.compute(update_history=False)
 if list(obs['actor'].shape)!=[1,616]:raise RuntimeError('Unexpected policy observation dimension')
 natural=[c['joint_names'].index(n) for n in robot.joint_names]
 if len(natural)!=29 or set(robot.joint_names)!=set(c['joint_names']):raise RuntimeError('G1 joint set mismatch')
 inverse=mt.policy_joint_ids;rows=[]
 action_term=env.action_manager.get_term('joint_pos')
 if list(action_term.target_names)!=list(robot.joint_names):raise RuntimeError('Action target ordering differs from natural joint order: '+repr((action_term.target_names,robot.joint_names)))
 for name,actual in [('action_scale',action_term.scale),('action_offset',action_term.offset)]:
  expected=torch.tensor(c[name],device=env.device)[natural]
  if not torch.equal(actual[0],expected):raise RuntimeError('Action contract mismatch: '+name)
 np.savez_compressed(a.output/'policy_io_private.npz',observations=obs['actor'].cpu().numpy(),actions=policy(obs['actor']).cpu().numpy())
 receipt['action_scale_offset_exact']=True
 original=env.termination_manager.compute
 def capture():
  # mjlab's default pre-termination body kinematics lag one substep. Refresh
  # derived state to match the original post-physics/pre-reset protocol.
  env.sim.forward()
  result=original()
  rows.append({'frame':int(mt.time_steps[0]),'reference_body_pos':mt.body_pos_w[0].detach().cpu().numpy().copy(),
   'robot_body_pos':mt.robot_body_pos_w[0].detach().cpu().numpy().copy(),'reference_joint_pos':mt.joint_pos[0,inverse].detach().cpu().numpy().copy(),
   'robot_joint_pos':robot.data.joint_pos[0,inverse].detach().cpu().numpy().copy(),'reference_joint_vel':mt.joint_vel[0,inverse].detach().cpu().numpy().copy(),
   'robot_joint_vel':robot.data.joint_vel[0,inverse].detach().cpu().numpy().copy(),'terminated':bool(env.termination_manager.terminated[0]),'timeout':bool(env.termination_manager.time_outs[0])})
  return result
 env.termination_manager.compute=capture
 started=time.monotonic();limit=a.max_steps or nframes+2;ended=False
 for step in range(limit):
  actions=policy(obs['actor'])[:,natural]
  obs,_,terminated,timeout,_=env.step(actions)
  if bool(terminated[0]) or bool(timeout[0]):ended=True;break
 raw={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
 for key,value in raw.items():
  if not np.isfinite(value).all():raise RuntimeError('Nonfinite recorded state: '+key)
 if not np.array_equal(raw['frame'],np.arange(len(rows))):raise RuntimeError('Reference frame continuity mismatch')
 np.savez_compressed(a.output/'raw_pre_reset.npz',**raw)
 metrics={'mpjpe_m_until_termination':float(np.linalg.norm(raw['reference_body_pos']-raw['robot_body_pos'],axis=-1).mean()),
  'mpjae_rad_until_termination':float(np.abs(raw['reference_joint_pos']-raw['robot_joint_pos']).mean()),
  'mpjve_rad_s_until_termination':float(np.abs(raw['reference_joint_vel']-raw['robot_joint_vel']).mean())}
 model=env.sim.mj_model;body_mass={n:float(model.body_mass[robot.indexing.body_ids[i]]) for i,n in enumerate(robot.body_names)}
 shared={n:{'isaac':c['body_mass'][c['body_names'].index(n)],'mjlab':body_mass[n]} for n in c['tracked_body_names']}
 receipt.update(status='FIRST_EPISODE_EVALUATED' if ended else 'BOUNDED_PHYSICS_PROBE_COMPLETED',
  frames_evaluated=len(rows),terminated=bool(raw['terminated'][-1]),timeout=bool(raw['timeout'][-1]),
  full_clip_success=bool(ended and len(rows)==nframes and not raw['terminated'].any()),metrics=metrics,
  elapsed_rollout_seconds=time.monotonic()-started,physics_steps=len(rows)*cfg.decimation,control_dt=env.step_dt,physics_dt=env.physics_dt,
  compiled_contact_parameters=[{'name':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i),'friction':model.geom_friction[i].tolist(),'condim':int(model.geom_condim[i]),'priority':int(model.geom_priority[i])} for i in range(model.ngeom) if 'foot' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '') or 'terrain' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '')],
  executed_sources_unchanged=verify_sources(receipt['source_provenance']),
  post_physics_forward_before_termination=True,body_mass_comparison=shared,
  note='Errors censored at first termination; interpret with duration and success. Default framework geometry/solver differences are retained and disclosed; same PD/action/command contract. No new policy optimization.',
  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 save();print('UNITREE_MJLAB_PHYGILE_RESULT',json.dumps(receipt),flush=True)
except BaseException as exc:
 receipt.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());save();raise
finally:
 if env is not None:env.close()

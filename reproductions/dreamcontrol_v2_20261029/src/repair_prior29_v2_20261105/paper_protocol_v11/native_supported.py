"""Task-specific PhysX geometry/contact probes of planned generated references.

Robot and object poses are explicitly injected. This certifies the stated scene
geometry and source/FK bindings, never PD tracking, grasping or task success.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=['pour','bimanual_box_lift','box_push_pull'],required=True)
p.add_argument('--candidate-dir',action='append',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();args.output_dir.mkdir(exist_ok=False);app=AppLauncher(args).app
import torch
from pxr import Usd,UsdGeom,UsdPhysics
import isaaclab.sim as sim
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv
from stage3_pick29_env import Pick29EnvCfg,ASSET
from stage3_pick29_core import BODY_JOINTS,sha

def main():
 torch.cuda.set_per_process_memory_fraction(.10,device=args.device)
 sources=[];reports=[];scene=None;scene_sha=None
 for directory in args.candidate_dir:
  r=json.loads((directory/'report.json').read_text());assert r['passed_requested_kinematic_checks'] and r['task']==args.task
  path=directory/'robot29.npz';assert sha(path)==r['output_sha256'];assert sha(r['sample'])==r['sample_sha256']
  s=dict(np.load(path,allow_pickle=False));assert s['joint_names'].tolist()==BODY_JOINTS and s['q'].shape==(180,29) and float(s['fps'])==20.
  assert np.allclose(s['sample_times'],np.arange(180)/20) and 'planned_object_positions' in s
  source_scene=json.loads((Path(r['sample']).parent/'report.json').read_text())['scene']
  h=hashlib.sha256(json.dumps(source_scene,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert h==r['scene_sha256']
  if scene is None:scene=source_scene;scene_sha=h
  else:assert source_scene==scene and scene_sha==h
  sources.append(s);reports.append(r)
 cfg=Pick29EnvCfg();cfg.scene.num_envs=len(sources);cfg.sim.device=args.device;cfg.seed=20261104;cfg.reference_manifest=''
 cfg.scene.robot.spawn.rigid_props.disable_gravity=True
 cfg.scene.table.spawn.size=tuple(scene['table']['size']);cfg.scene.table.init_state.pos=tuple(scene['table']['center'])
 cfg.scene.object.spawn.size=tuple(scene['object_size']);cfg.scene.object.init_state.pos=tuple(scene['nominal_object_center'])
 cfg.scene.object.spawn.rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True)
 if args.task=='pour':
  cfg.scene.receiver=AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Receiver',init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(scene['receiver_center'])),
   spawn=sim.CuboidCfg(size=tuple(scene['receiver_size']),collision_props=sim.CollisionPropertiesCfg(),visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.6,.3,.1))))
 env=ManagerBasedRLEnv(cfg=cfg)
 try:
  env.reset();robot=env.scene['robot'];obj=env.scene['object'];sensor=env.scene['contact_forces'];device=env.device
  ids=robot.find_joints(BODY_JOINTS,preserve_order=True)[0];assert len(ids)==29 and len(robot.joint_names)==43
  wrists=robot.find_bodies(['left_wrist_yaw_link','right_wrist_yaw_link'],preserve_order=True)[0];assert len(wrists)==2
  active=[1] if args.task=='pour' else [0,1]
  usd=Usd.Stage.Open(str(ASSET));cache=UsdGeom.XformCache();spheres=[]
  for prim in Usd.PrimRange(usd.GetDefaultPrim(),Usd.TraverseInstanceProxies()):
   if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
   path=str(prim.GetPath())
   if not any('/'+side+'_ankle_roll_link/' in path for side in ['left','right']):continue
   assert prim.IsA(UsdGeom.Sphere);side='left' if '/left_ankle_roll_link/' in path else 'right'
   body=usd.GetPrimAtPath(str(usd.GetDefaultPrim().GetPath())+'/'+side+'_ankle_roll_link');local=cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse()
   scales=np.linalg.norm(np.asarray(local)[:3,:3],axis=1);assert np.max(np.abs(scales-scales[0]))<1e-6
   spheres.append({'body':side+'_ankle_roll_link','center':list(local.ExtractTranslation()),'radius':float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()*scales[0])})
  assert len(spheres)==8
  sensor_ids=[k for k,name in enumerate(sensor.body_names) if not ('ankle_' in name or '_hand_' in name or '_wrist_' in name)]
  maximum=lambda:torch.zeros(env.num_envs,device=device)
  max_force=maximum();max_wrist=maximum();max_orientation=maximum();max_object_error=maximum()
  min_height=torch.full((env.num_envs,),float('inf'),device=device);max_sole=-min_height.clone();min_clearance=min_height.clone();min_object_clearance=min_height.clone()
  max_bodies=torch.zeros(env.num_envs,len(sensor_ids),device=device)
  object_state=obj.data.default_root_state.clone();object_state[:,7:]=0;object_state[:,3:7]=torch.tensor([1.,0.,0.,0.],device=device)
  origin=env.scene.env_origins;began=time.monotonic()
  corners=torch.tensor(np.array([[a,b,c] for a in [-1,1] for b in [-1,1] for c in [-1,1]])*np.array(scene['object_size'])/2,device=device,dtype=torch.float32)
  table_center=torch.tensor(scene['table']['center'],device=device);extent=torch.tensor(scene['table']['size'],device=device)/2;table_top=float(scene['table']['center'][2]+scene['table']['size'][2]/2)
  reset_error=None
  for frame in range(180):
   object_state[:,:3]=torch.tensor(np.stack([s['planned_object_positions'][frame] for s in sources]),device=device)+origin
   if args.task=='pour':
    object_state[:,3:7]=torch.tensor(np.stack([s['planned_object_quaternions_wxyz'][frame] for s in sources]),device=device)
   obj.write_root_state_to_sim(object_state)
   q=torch.tensor(np.stack([s['q'][frame] for s in sources]),device=device)
   root=torch.tensor(np.stack([np.r_[s['root_pos'][frame],s['root_quat'][frame],np.zeros(6)] for s in sources]),device=device,dtype=torch.float32);root[:,:3]+=origin
   all_q=robot.data.default_joint_pos.clone();all_q[:,ids]=q
   hands=torch.tensor(np.stack([s['hand_closed'][frame] for s in sources]),device=device);action=torch.zeros(env.num_envs,31,device=device);action[:,-2:]=1-2*hands
   env.action_manager.process_action(action)
   for term in ['left_hand','right_hand']:
    h=env.action_manager.get_term(term);all_q[:,h._joint_ids]=h.processed_actions
   robot.write_root_state_to_sim(root);robot.write_joint_state_to_sim(all_q,torch.zeros_like(all_q));robot.set_joint_position_target(all_q)
   env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.physics_dt)
   error=(obj.data.root_pos_w-object_state[:,:3]).norm(dim=-1);max_object_error=torch.maximum(max_object_error,error)
   if frame==0:reset_error=float(error.max())
   wrist=robot.data.body_pos_w[:,wrists]-origin[:,None];expected=torch.tensor(np.stack([s['keypoints'][frame,20:22] for s in sources]),device=device)
   max_wrist=torch.maximum(max_wrist,(wrist-expected).norm(dim=-1).amax(dim=1))
   rotations=torch.tensor(np.stack([np.stack([s['left_wrist_quat'][frame],s['right_wrist_quat'][frame]]) for s in sources]),device=device)
   max_orientation=torch.maximum(max_orientation,math_utils.quat_error_magnitude(robot.data.body_quat_w[:,wrists],rotations).amax(dim=1))
   distance=(wrist[:,active]-table_center).abs()-extent;sdf=distance.clamp_min(0).norm(dim=-1)+distance.amax(dim=-1).clamp_max(0)
   min_clearance=torch.minimum(min_clearance,sdf.amin(dim=1))
   object_corners=math_utils.quat_apply(object_state[:,None,3:7].expand(-1,8,-1),corners[None].expand(env.num_envs,-1,-1))
   bottom=object_corners[:,:,2].amin(dim=1)+object_state[:,2]-origin[:,2]-table_top;min_object_clearance=torch.minimum(min_object_clearance,bottom)
   foot_low=[[],[]]
   for sphere in spheres:
    bi=robot.find_bodies(sphere['body'])[0][0];center=torch.tensor(sphere['center'],device=device,dtype=torch.float32).expand(env.num_envs,-1)
    point=robot.data.body_pos_w[:,bi]+math_utils.quat_apply(robot.data.body_quat_w[:,bi],center)
    foot_low[0 if sphere['body'].startswith('left') else 1].append(point[:,2]-origin[:,2]-sphere['radius'])
   soles=torch.stack([torch.stack(x).amin(dim=0) for x in foot_low],dim=1);min_height=torch.minimum(min_height,soles.amin(dim=1));max_sole=torch.maximum(max_sole,soles.amax(dim=1))
   env.sim.step(render=False);env.scene.update(env.physics_dt)
   forces=sensor.data.net_forces_w[:,sensor_ids].norm(dim=-1);max_bodies=torch.maximum(max_bodies,forces);max_force=torch.maximum(max_force,forces.amax(dim=1))
  results=[];refs=[]
  for k,r in enumerate(reports):
   checks={'ground_no_penetration':float(min_height[k])>=-.005,'actual_two_wrist_fk_consistency':float(max_wrist[k])<1e-4,
    'actual_two_wrist_rotation_consistency':float(max_orientation[k])<1e-4,'active_wrist_table_clearance_03m':float(min_clearance[k])>.03,
    'planned_object_pose_binding':float(max_object_error[k])<1e-5,'planned_object_table_no_penetration':float(min_object_clearance[k])>=-.005}
   accepted=all(checks.values());proof=args.output_dir/('candidate_'+str(r['candidate_id'])+'_admission.json')
   record={'candidate_id':r['candidate_id'],'task':args.task,'scene_sha256':scene_sha,'asset_sha256':sha(ASSET),'source_state_sha256':r['output_sha256'],
    'kinematic_report_sha256':sha(args.candidate_dir[k]/'report.json'),'native_replay_frames':180,'source_fps':20,'native_checks':checks,
    'eligible_for_calibrated_scene_reference':accepted,'minimum_native_foot_height_m':float(min_height[k]),'maximum_native_foot_height_m':float(max_sole[k]),
    'max_non_hand_non_foot_force_N':float(max_force[k]),'wrist_fk_max_error_m':float(max_wrist[k]),'wrist_rotation_max_error_rad':float(max_orientation[k]),
    'minimum_active_wrist_table_clearance_m':float(min_clearance[k]),'planned_object_reset_error_m':reset_error,
    'minimum_planned_object_table_clearance_m':float(min_object_clearance[k]),
    'largest_contact_bodies':[{'body':sensor.body_names[sensor_ids[j]],'max_force_N':float(max_bodies[k,j])} for j in torch.argsort(max_bodies[k],descending=True)[:5].tolist()],
    'object_motion_policy':'explicitly injected planned position/orientation; no contact-driven object motion claim',
    'scope':'task-specific pose-injected geometry/contact applicability; one0.005s physics contact probe per20Hz reference state, not timed tracking or task success'}
   proof.open('x').write(json.dumps(record,indent=2,allow_nan=False));results.append(record)
   if accepted:refs.append({'candidate_id':r['candidate_id'],'task':args.task,'path':r['output'],'sha256':r['output_sha256'],'kinematic_report':str(args.candidate_dir[k]/'report.json'),'native_proof':str(proof),'native_proof_sha256':sha(proof),'scene_sha256':scene_sha})
  (args.output_dir/'reference_bank.json').open('x').write(json.dumps({'task':args.task,'body_dof':29,'reference_fps':20,'asset_sha256':sha(ASSET),'scene':scene,'scene_sha256':scene_sha,'references':refs,'scope':'task-scene reference admission, not dynamic tracking success'},indent=2))
  report={'status':'native_task_replay_completed','task':args.task,'candidates':len(reports),'admitted_references':len(refs),'frames':180,'elapsed_s':time.monotonic()-began,'results':results,'formal_training_started':False}
  (args.output_dir/'report.json').open('x').write(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
 finally:env.close()
try:main()
finally:app.close()

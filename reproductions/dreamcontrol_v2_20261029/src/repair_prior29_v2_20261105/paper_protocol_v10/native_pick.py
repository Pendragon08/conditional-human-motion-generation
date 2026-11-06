"""Native PhysX collision replay of generated candidates; no policy updates.

Poses are injected to test geometry/contact, not to claim PD tracking or Pick
success. Accepted references bind native results, the asset and the scene.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--candidate-dir',action='append',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
args.output_dir.mkdir(exist_ok=False)
app=AppLauncher(args).app
import torch
from pxr import Usd,UsdGeom,UsdPhysics
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from stage3_pick29_env import Pick29EnvCfg,ASSET
from stage3_pick29_core import BODY_JOINTS,SCENE,SCENE_SHA,sha


def main():
    torch.cuda.set_per_process_memory_fraction(.12,device=args.device)
    config=Pick29EnvCfg();config.scene.num_envs=len(args.candidate_dir);config.sim.device=args.device
    config.scene.robot.spawn.rigid_props.disable_gravity=True
    config.seed=20261103
    env=ManagerBasedRLEnv(cfg=config)
    try:
        env.reset();robot=env.scene['robot'];sensor=env.scene['contact_forces']
        ids=robot.find_joints(BODY_JOINTS,preserve_order=True)[0]
        sources=[];checks=[];reports=[]
        for directory in args.candidate_dir:
            report=json.loads((directory/'report.json').read_text())
            assert report['passed_requested_kinematic_checks'] and report['scene_sha256']==SCENE_SHA
            path=directory/'robot29.npz';assert sha(path)==report['output_sha256']
            with np.load(path,allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
            assert source['joint_names'].tolist()==BODY_JOINTS and float(source['fps'])==20.
            assert 'task_object_position' in source and 'task_object_quaternion_wxyz' in source
            assert np.allclose(source['task_object_position'],report['task_object_position'],atol=1e-12)
            assert np.allclose(source['task_object_quaternion_wxyz'],report['task_object_quaternion_wxyz'],atol=1e-12)
            sources.append(source);reports.append(report)
        assert len({len(s['q']) for s in sources})==1
        frames=len(sources[0]['q'])
        obj=env.scene['object']
        object_state=obj.data.default_root_state.clone()
        object_state[:,:3]=torch.tensor(np.stack([x['task_object_position'] for x in sources]),device=env.device)+env.scene.env_origins
        object_state[:,3:7]=torch.tensor(np.stack([x['task_object_quaternion_wxyz'] for x in sources]),device=env.device)
        object_state[:,7:]=0.
        obj.write_root_state_to_sim(object_state)
        env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.physics_dt)
        object_reset_error=float((obj.data.root_pos_w-object_state[:,:3]).abs().max())
        assert object_reset_error<1e-5,'per-reference object placement mismatch'
        # Read the actual four foot-sphere colliders from the instance proxies,
        # composing every USD transform into the ankle frame. These are not
        # body-center or bounding-box substitutes for native floor geometry.
        usd=Usd.Stage.Open(str(ASSET));cache=UsdGeom.XformCache()
        spheres=[]
        for prim in Usd.PrimRange(usd.GetDefaultPrim(),Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            path=str(prim.GetPath())
            if not any('/'+side+'_ankle_roll_link/' in path for side in ('left','right')):continue
            assert prim.IsA(UsdGeom.Sphere),'unexpected foot collider '+path
            side='left' if '/left_ankle_roll_link/' in path else 'right'
            body=usd.GetPrimAtPath(str(usd.GetDefaultPrim().GetPath())+'/'+side+'_ankle_roll_link')
            local=cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse()
            scales=np.linalg.norm(np.asarray(local)[:3,:3],axis=1)
            assert np.max(np.abs(scales-scales[0]))<1e-6
            spheres.append({'body':side+'_ankle_roll_link','center':list(local.ExtractTranslation()),
                            'radius':float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()*scales[0]),'path':path})
        assert len(spheres)==8,'expected native four foot spheres per side'
        sensor_ids=[k for k,name in enumerate(sensor.body_names) if not
                    ('ankle_' in name or '_hand_' in name or '_wrist_' in name)]
        max_forces=torch.zeros(env.num_envs,device=env.device)
        max_body_forces=torch.zeros(env.num_envs,len(sensor_ids),device=env.device)
        min_height=torch.full((env.num_envs,),float('inf'),device=env.device)
        max_sole=torch.full((env.num_envs,),-float('inf'),device=env.device)
        max_wrist_error=torch.zeros(env.num_envs,device=env.device)
        min_wrist_clearance=torch.full((env.num_envs,),float('inf'),device=env.device)
        wrist_id=robot.find_bodies('right_wrist_yaw_link')[0][0]
        began=time.monotonic()
        for frame in range(frames):
            q=torch.tensor(np.stack([s['q'][frame] for s in sources]),device=env.device)
            root=torch.tensor(np.stack([np.r_[s['root_pos'][frame],s['root_quat'][frame],np.zeros(6)] for s in sources]),device=env.device,dtype=torch.float32)
            root[:,:3]+=env.scene.env_origins
            all_q=robot.data.default_joint_pos.clone();all_q[:,ids]=q
            hands=torch.tensor(np.stack([s['hand_closed'][frame] for s in sources]),device=env.device)
            action=torch.zeros(env.num_envs,31,device=env.device);action[:,-2:]=1-2*hands
            env.action_manager.process_action(action)
            for term in ['left_hand','right_hand']:
                hand=env.action_manager.get_term(term)
                all_q[:,hand._joint_ids]=hand.processed_actions
            robot.write_root_state_to_sim(root)
            robot.write_joint_state_to_sim(all_q,torch.zeros_like(all_q))
            robot.set_joint_position_target(all_q)
            env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.physics_dt)
            wrist=robot.data.body_pos_w[:,wrist_id]-env.scene.env_origins
            intended=torch.tensor(np.stack([s['keypoints'][frame,21] for s in sources]),device=env.device)
            max_wrist_error=torch.maximum(max_wrist_error,(wrist-intended).norm(dim=-1))
            center=torch.tensor(SCENE['table_center'],device=env.device);extent=torch.tensor(SCENE['table_size'],device=env.device)/2
            distances=(wrist-center).abs()-extent
            sdf=distances.clamp_min(0).norm(dim=-1)+distances.amax(dim=-1).clamp_max(0)
            min_wrist_clearance=torch.minimum(min_wrist_clearance,sdf)
            foot_low=[[],[]]
            for sphere in spheres:
                bi=robot.find_bodies(sphere['body'])[0][0]
                center=torch.tensor(sphere['center'],device=env.device,dtype=torch.float32).expand(env.num_envs,-1)
                point=robot.data.body_pos_w[:,bi]+math_utils.quat_apply(robot.data.body_quat_w[:,bi],center)
                height=point[:,2]-env.scene.env_origins[:,2]-sphere['radius']
                foot_low[0 if sphere['body'].startswith('left') else 1].append(height)
            soles=torch.stack([torch.stack(side).amin(dim=0) for side in foot_low],dim=1)
            min_height=torch.minimum(min_height,soles.amin(dim=1));max_sole=torch.maximum(max_sole,soles.amax(dim=1))
            # One real physics step computes native contact manifolds. Pose is
            # reinjected next frame: this is deliberately not a tracking test.
            env.sim.step(render=False);env.scene.update(env.physics_dt)
            body_forces=sensor.data.net_forces_w[:,sensor_ids].norm(dim=-1)
            max_body_forces=torch.maximum(max_body_forces,body_forces)
            forces=body_forces.amax(dim=1)
            max_forces=torch.maximum(max_forces,forces)
        bank=[]
        calibration=ASSET.parents[1]/'reports/physx_smoke_128_aligned/status.json'
        for k,directory in enumerate(args.candidate_dir):
            native={'ground_no_penetration':float(min_height[k])>=-.005,
                    'actual_wrist_fk_consistency':float(max_wrist_error[k])<1e-4,
                    'wrist_table_clearance_03m':float(min_wrist_clearance[k])>.03}
            accepted=all(native.values()) and reports[k]['passed_requested_kinematic_checks']
            result={'task_object_position':sources[k]['task_object_position'].tolist(),
                    'task_object_quaternion_wxyz':sources[k]['task_object_quaternion_wxyz'].tolist(),
                    'per_reference_object_reset_error_m':object_reset_error,
                    'eligible_for_calibrated_scene_reference':accepted,'kinematic_report_sha256':sha(directory/'report.json'),
                    'native_checks':native,'asset_sha256':sha(ASSET),'scene_sha256':SCENE_SHA,
                    'native_replay_frames':frames,'minimum_native_foot_sphere_height_m':float(min_height[k]),
                    'maximum_native_foot_sphere_height_m':float(max_sole[k]),'max_non_hand_non_foot_contact_force_N':float(max_forces[k]),
                    'minimum_wrist_table_clearance_m':float(min_wrist_clearance[k]),
                    'actual_wrist_position_max_error_m':float(max_wrist_error[k]),
                    'largest_contact_bodies':[{'body':sensor.body_names[sensor_ids[j]],'max_force_N':float(max_body_forces[k,j])}
                                               for j in torch.argsort(max_body_forces[k],descending=True)[:5].tolist()],
                    'scope':'pose-injected native contact/geometry replay; no PD tracking or grasp/task-success claim'}
            result_path=args.output_dir/('candidate_%d_admission.json'%k)
            with result_path.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
            checks.append(result)
            if accepted:bank.append({'path':str(directory/'robot29.npz'),'sha256':sha(directory/'robot29.npz'),
                                    'postopt_report':str(result_path),'postopt_report_sha256':sha(result_path),
                                    'kinematic_report':str(directory/'report.json'),'grasp_time_s':2.,
                                    'task_object_position':sources[k]['task_object_position'].tolist(),
                                    'task_object_quaternion_wxyz':sources[k]['task_object_quaternion_wxyz'].tolist()})
        if len(bank)>=2:
            manifest={'asset_sha256':sha(ASSET),'scene_sha256':SCENE_SHA,'references':bank,
                      'object_pose_policy':'per_reference_fixed','task':'pick','minimum_formal_reference_count':500,
                      'calibration_report':str(calibration),'calibration_report_sha256':sha(calibration)}
            with (args.output_dir/'reference_bank.json').open('x') as f:json.dump(manifest,f,indent=2)
        result={'status':'native_replay_completed','frames':frames,'candidates':len(sources),'admitted_references':len(bank),
                'PPO_started':False,'elapsed_s':time.monotonic()-began,'candidates_results':checks,'native_foot_spheres':spheres}
        with (args.output_dir/'report.json').open('x') as f:json.dump(result,f,indent=2)
        print(json.dumps(result),flush=True)
    finally:env.close()


try:main()
finally:app.close()

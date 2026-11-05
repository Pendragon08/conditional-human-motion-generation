"""Separate live Pour, bimanual carry/place and push/release/pull PhysX tasks.

Only robot states initialize from references. Objects remain dynamic and are
never moved along planned paths. Common G1 actuation is reused read-only.
"""
import json
from pathlib import Path
import torch
import isaaclab.sim as sim
import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg,ObservationGroupCfg,ObservationTermCfg,RewardTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
import stage3_pick29_env as base
from stage3_pick29_core import BODY_JOINTS,FixedFrameMap,sha
from stage3_supported_ppo_core_v1 import TaskReferenceBank,empty_milestones,update_milestones

def startup(env,env_ids=None):
    env._reference_bank=TaskReferenceBank(env.cfg.reference_manifest,env.cfg.skill_task,
                                         sha(base.ASSET),env.device,env.cfg.minimum_reference_count)
    env._reference_ids=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
    env._reference_cache_step=-1;env._task_cache_step=-1
    env._body29_ids=env.scene['robot'].find_joints(BODY_JOINTS,preserve_order=True)[0]
    env._keypoint_frames=FixedFrameMap(base.URDF,env.scene['robot'].body_names,base.KEYPOINT_LINKS,env.device)
    env._wrists=env.scene['robot'].find_bodies(['left_wrist_yaw_link','right_wrist_yaw_link'],preserve_order=True)[0]
    env._milestones=empty_milestones(env.num_envs,env.device)
    env._last_episode=empty_milestones(env.num_envs,env.device)
    env._object_initial=torch.zeros(env.num_envs,3,device=env.device)
    env._contact_tensor=torch.zeros(env.num_envs,2,device=env.device,dtype=torch.bool)

def reset_scene(env,env_ids):
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    for key in env._milestones:
        env._last_episode[key][env_ids]=env._milestones[key][env_ids]
        env._milestones[key][env_ids]=False
    base.reset_scene(env,env_ids)
    bank=env._reference_bank;selected=env._reference_ids[env_ids]
    obj=env.scene['object'];state=obj.data.default_root_state[env_ids].clone()
    env._object_initial[env_ids]=bank.object_positions[selected]
    if env.cfg.fresh_evaluation_positions:
        center=torch.tensor(bank.scene['nominal_object_center'],device=env.device)
        env._object_initial[env_ids]=center
        env._object_initial[env_ids,:2]+=torch.rand(len(env_ids),2,device=env.device)*.05-.025
    state[:,:3]=env._object_initial[env_ids]+env.scene.env_origins[env_ids]
    state[:,3:7]=bank.object_quaternions[selected];state[:,7:]=0
    obj.write_root_state_to_sim(state,env_ids=env_ids)
    env._task_cache_step=-1

def contact_pairs(env):
    result=[]
    for side in ['left','right']:
        force=[]
        for finger in ['index','middle','thumb']:
            value=env.scene[side+'_'+finger+'_object_contact'].data.force_matrix_w
            assert value is not None,'object-filtered finger contact unavailable'
            force.append(value.reshape(env.num_envs,-1,3).norm(dim=-1).amax(-1))
        result.append(torch.stack(force).amax(0)>.5)
    return torch.stack(result,dim=-1)

def measured(env):
    if env._task_cache_step==env.common_step_counter:return env._task_measured
    robot=env.scene['robot'];obj=env.scene['object'];pos=obj.data.root_pos_w-env.scene.env_origins
    wrists=robot.data.body_pos_w[:,env._wrists]-env.scene.env_origins[:,None]
    # Distance to the current, oriented object's surface. No reference targets.
    local=math_utils.quat_apply_inverse(obj.data.root_quat_w[:,None].expand(-1,2,-1),wrists-pos[:,None])
    extent=torch.tensor(env._reference_bank.scene['object_size'],device=env.device)/2
    outside=(local.abs()-extent).clamp_min(0);gaps=outside.norm(dim=-1)
    contacts=contact_pairs(env);closed=env.action_manager.action[:,-2:]<0
    z=torch.zeros(env.num_envs,3,device=env.device);z[:,2]=1
    axis=math_utils.quat_apply(obj.data.root_quat_w,z)
    tilt=torch.acos(axis[:,2].clamp(-1,1));speed=obj.data.root_lin_vel_w.norm(dim=-1)
    seconds=env.episode_length_buf*env.step_dt
    update_milestones(env.cfg.skill_task,env._milestones,seconds,pos,env._object_initial,
                      tilt,speed,contacts,closed,gaps)
    env._task_measured={'position':pos,'delta':pos-env._object_initial,'tilt':tilt,
                        'contacts':contacts,'gaps':gaps,'seconds':seconds,'speed':speed}
    env._task_cache_step=env.common_step_counter;return env._task_measured

def task_reward(env):
    m=measured(env);f=env._milestones;t=env.cfg.skill_task;delta=m['delta']
    approach=torch.exp(-40*m['gaps'].square())
    if t=='pour':
        lift=(delta[:,2]/.16).clamp(0,1)*f['contact']
        tilt=(m['tilt']/(torch.pi*75/180)).clamp(0,1)*f['lifted']
        return (.2*approach[:,1]+lift+2*tilt+.5*f['upright']+3*f['success']).float()
    if t=='bimanual_box_lift':
        lift=(delta[:,2]/.14).clamp(0,1)*f['contact']
        carry=(delta[:,0]/.10).clamp(0,1)*f['lifted']
        return (.2*approach.mean(-1)+lift+2*carry+3*f['success']).float()
    push=(delta[:,0]/.10).clamp(0,1)*f['contact']
    returned=(1-delta[:,0].abs()/.10).clamp(0,1)*f['recontacted']
    return (.2*approach.mean(-1)+push+f['released']+2*returned+3*f['success']).float()

def wrists_orientation(env):
    ref=base.reference(env);robot=env.scene['robot']
    expected=torch.stack([ref['left_wrist_quat'],ref['right_wrist_quat']],dim=1)
    error=math_utils.quat_error_magnitude(robot.data.body_quat_w[:,env._wrists],expected)
    return error[:,1] if env.cfg.skill_task=='pour' else error.mean(-1)

def sensor(side,finger):
    suffix='2' if finger=='thumb' else '1'
    return ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+side+'_hand_'+finger+'_'+suffix+'_link',
          history_length=1,filter_prim_paths_expr=['{ENV_REGEX_NS}/Object'])

@configclass
class SupportedSceneCfg(base.SceneCfg):
    left_index_object_contact= sensor('left','index')
    left_middle_object_contact=sensor('left','middle')
    left_thumb_object_contact=sensor('left','thumb')
    right_index_object_contact=sensor('right','index')
    right_middle_object_contact=sensor('right','middle')
    right_thumb_object_contact=sensor('right','thumb')

@configclass
class EventsCfg(base.EventsCfg):
    initialize=EventTermCfg(func=startup,mode='startup')
    reset=EventTermCfg(func=reset_scene,mode='reset')

@configclass
class RewardsCfg(base.RewardsCfg):
    pick=None
    wrist_orientation=RewardTermCfg(func=wrists_orientation,weight=-1.)
    interaction=RewardTermCfg(func=task_reward,weight=1.)

@configclass
class SupportedEnvCfg(base.Pick29EnvCfg):
    scene:SupportedSceneCfg=SupportedSceneCfg(num_envs=4,env_spacing=3.)
    events:EventsCfg=EventsCfg()
    rewards:RewardsCfg=RewardsCfg()
    skill_task:str=''
    minimum_reference_count:int=500
    fresh_evaluation_positions:bool=False

def make_cfg(task,manifest,device,num_envs,minimum=500,evaluation=False):
    m=json.loads(Path(manifest).read_text());scene=m['scene'];assert m['task']==task
    cfg=SupportedEnvCfg();cfg.skill_task=task;cfg.reference_manifest=str(manifest)
    cfg.minimum_reference_count=minimum;cfg.sim.device=device;cfg.scene.num_envs=num_envs
    cfg.seed=20261105;cfg.fresh_evaluation_positions=evaluation
    cfg.scene.table.spawn.size=tuple(scene['table']['size']);cfg.scene.table.init_state.pos=tuple(scene['table']['center'])
    cfg.scene.object.spawn.size=tuple(scene['object_size'])
    cfg.scene.object.init_state.pos=tuple(scene['nominal_object_center'])
    cfg.scene.object.spawn.rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=False,disable_gravity=False)
    cfg.scene.object.spawn.mass_props=sim.MassPropertiesCfg(mass=.1 if task=='pour' else 1.)
    if task=='pour':
        cfg.scene.receiver=AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Receiver',
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(scene['receiver_center'])),
            spawn=sim.CuboidCfg(size=tuple(scene['receiver_size']),collision_props=sim.CollisionPropertiesCfg(),
                               visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.6,.3,.1))))
    return cfg

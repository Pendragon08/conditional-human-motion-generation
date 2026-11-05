"""Isolated Pick adaptation: G1 29 body joints, two binary hands, live actor inputs.

Import only after AppLauncher has started Isaac Sim. Original task/assets are read-only.
"""
from pathlib import Path
import torch

import isaaclab.sim as sim
import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.managers import RewardTermCfg, TerminationTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets import G1_CFG

from stage3_pick29_core import BODY_JOINTS, SCENE, SCENE_SHA, ReferenceBank29, FixedFrameMap, actor_observation, sha

D = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_pick29_20261103')
ASSET = D/'assets/g1_29dof_with_hands_urdf_aligned.usd'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
KEYPOINT_LINKS = ['pelvis_contour_link','left_hip_pitch_link','right_hip_pitch_link',
    'imu_link','left_knee_link','right_knee_link','torso_link','left_ankle_pitch_link',
    'right_ankle_pitch_link','imu_link','left_ankle_roll_link','right_ankle_roll_link',
    'mid360_link','left_shoulder_yaw_link','right_shoulder_yaw_link','d435_link',
    'left_shoulder_pitch_link','right_shoulder_pitch_link','left_elbow_link','right_elbow_link',
    'left_wrist_yaw_link','right_wrist_yaw_link']


def body_ids(env):
    if not hasattr(env, '_body29_ids'):
        env._body29_ids = env.scene['robot'].find_joints(BODY_JOINTS, preserve_order=True)[0]
    return env._body29_ids


def live_observation(env):
    robot,obj = env.scene['robot'],env.scene['object']
    ids = body_ids(env)
    return actor_observation(robot.data.root_ang_vel_b, robot.data.projected_gravity_b,
        robot.data.joint_pos[:,ids]-robot.data.default_joint_pos[:,ids],
        robot.data.joint_vel[:,ids]-robot.data.default_joint_vel[:,ids], env.action_manager.action,
        robot.data.root_pos_w,robot.data.root_quat_w,obj.data.root_pos_w,obj.data.root_quat_w)


def startup(env, env_ids=None):
    env._reference_bank = (ReferenceBank29(env.cfg.reference_manifest, sha(ASSET), env.device)
                           if env.cfg.reference_manifest else None)
    env._reference_ids = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._reference_cache_step=-1
    env._body29_ids = env.scene['robot'].find_joints(BODY_JOINTS,preserve_order=True)[0]
    env._keypoint_frames = FixedFrameMap(URDF,env.scene['robot'].body_names,KEYPOINT_LINKS,env.device)
    if len(env._body29_ids) != 29:
        raise RuntimeError('PhysX did not instantiate all 29 body joints')


def reset_scene(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs,device=env.device)
    mdp.reset_scene_to_default(env,env_ids)
    bank = env._reference_bank
    env._reference_cache_step=-1
    if bank is not None:
        ids = torch.randint(bank.count,(len(env_ids),),device=env.device)
        env._reference_ids[env_ids] = ids
        ref = bank.query(ids,torch.zeros(len(env_ids),device=env.device))
        robot = env.scene['robot']
        root = robot.data.default_root_state[env_ids].clone()
        root[:,:3] = ref['root_pos']+env.scene.env_origins[env_ids]
        root[:,3:7] = ref['root_quat']; root[:,7:] = 0
        robot.write_root_state_to_sim(root,env_ids=env_ids)
        q = robot.data.default_joint_pos[env_ids].clone()
        q[:,body_ids(env)] = ref['q']
        robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=env_ids)


def reference(env):
    if env._reference_bank is None:
        return None
    if env._reference_cache_step!=env.common_step_counter:
        env._reference_cache=env._reference_bank.query(env._reference_ids,env.episode_length_buf*env.step_dt)
        env._reference_cache_step=env.common_step_counter
    return env._reference_cache


def tracking_error(env, kind):
    ref = reference(env)
    if ref is None:
        return torch.zeros(env.num_envs,device=env.device)
    robot = env.scene['robot']
    if kind == 'joints':
        return (robot.data.joint_pos[:,body_ids(env)]-ref['q']).abs().sum(-1)
    if kind == 'position':
        return (robot.data.root_pos_w-env.scene.env_origins-ref['root_pos']).norm(dim=-1)
    if kind == 'orientation':
        return math_utils.quat_error_magnitude(robot.data.root_quat_w,ref['root_quat'])
    if kind == 'hand':
        return ((env.action_manager.action[:,-2:]<0)==(ref['hand_closed']>.5)).float().mean(-1)
    if kind == 'wrist_orientation':
        idx = robot.find_bodies('right_wrist_yaw_link')[0][0]
        return math_utils.quat_error_magnitude(robot.data.body_quat_w[:,idx],ref['right_wrist_quat'])
    if kind == 'keypoints':
        points = env._keypoint_frames.positions(robot.data.body_pos_w,robot.data.body_quat_w)-env.scene.env_origins[:,None]
        return (points-ref['keypoints']).norm(dim=-1).sum(-1)
    raise ValueError(kind)


def pick_success(env):
    ref = reference(env)
    if ref is None:
        return torch.zeros(env.num_envs,device=env.device)
    height = env.scene['object'].data.root_pos_w[:,2]-env.scene.env_origins[:,2]
    return ((height>SCENE['success_height']) & ref['after_grasp']).float()


def fallen(env):
    robot = env.scene['robot']
    height = robot.data.root_pos_w[:,2]-env.scene.env_origins[:,2]
    return (height<.3) | (robot.data.projected_gravity_b[:,2]>-.5)


def foot_slide(env):
    robot=env.scene['robot']; sensor=env.scene['contact_forces']
    ri=robot.find_bodies('.*_ankle_roll_link')[0]
    si=sensor.find_bodies('.*_ankle_roll_link')[0]
    contact=sensor.data.net_forces_w_history[:,:,si].norm(dim=-1).amax(dim=1)>1.
    return (robot.data.body_lin_vel_w[:,ri,:2].norm(dim=-1)*contact).sum(-1)


def foot_tilt(env):
    robot=env.scene['robot'];ids=robot.find_bodies('.*_ankle_roll_link')[0]
    q=robot.data.body_quat_w[:,ids]
    z=torch.zeros(q.shape[:-1]+(3,),device=env.device);z[...,2]=1
    return (1-math_utils.quat_apply(q,z)[...,2]).square().sum(-1)


def reference_finished(env):
    if env._reference_bank is None:
        return torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    duration=(env._reference_bank.lengths[env._reference_ids]-1)/20
    return env.episode_length_buf*env.step_dt >= duration


ROBOT = G1_CFG.copy()
ROBOT.prim_path='{ENV_REGEX_NS}/Robot'
ROBOT.spawn.usd_path=str(ASSET)
ROBOT.init_state.joint_pos.update({'waist_roll_joint':0.,'waist_pitch_joint':0.,'.*_hand_.*':0.})
ROBOT.actuators['waist_extra']=ImplicitActuatorCfg(joint_names_expr=['waist_roll_joint','waist_pitch_joint'],
    effort_limit=50.,velocity_limit=22.67,stiffness=200.,damping=5.,armature=.01)


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path='/World/Ground',spawn=sim.GroundPlaneCfg())
    robot: ArticulationCfg = ROBOT
    table = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Table',
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(SCENE['table_center'])),
        spawn=sim.CuboidCfg(size=tuple(SCENE['table_size']), collision_props=sim.CollisionPropertiesCfg(),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.15,.3,.2)),
            physics_material=sim.RigidBodyMaterialCfg(static_friction=.9,dynamic_friction=.9)))
    object = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Object',
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(SCENE['object_center'])),
        spawn=sim.CuboidCfg(size=tuple(SCENE['object_size']),collision_props=sim.CollisionPropertiesCfg(),
            rigid_props=sim.RigidBodyPropertiesCfg(),mass_props=sim.MassPropertiesCfg(mass=SCENE['object_mass']),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.1,.2,.6)),
            physics_material=sim.RigidBodyMaterialCfg(static_friction=1.2,dynamic_friction=1.2)))
    contact_forces=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',history_length=3,track_air_time=True)
    light=AssetBaseCfg(prim_path='/World/Light',spawn=sim.DomeLightCfg(intensity=1000.,color=(.8,.8,.8)))


@configclass
class ActionsCfg:
    body = mdp.JointPositionActionCfg(asset_name='robot',joint_names=BODY_JOINTS,
                                     preserve_order=True,scale=.5,use_default_offset=True)
    left_hand = mdp.BinaryJointPositionActionCfg(asset_name='robot',joint_names=['left_hand.*'],
        open_command_expr={'left_hand.*':0.},close_command_expr={'left_hand_index_.*':-1.5707963,
            'left_hand_middle_.*':-1.5707963,'left_hand_thumb_0_joint':0.,
            'left_hand_thumb_1_joint':1.0471976,'left_hand_thumb_2_joint':1.5707963})
    right_hand = mdp.BinaryJointPositionActionCfg(asset_name='robot',joint_names=['right_hand.*'],
        open_command_expr={'right_hand.*':0.},close_command_expr={'right_hand_index_.*':1.5707963,
            'right_hand_middle_.*':1.5707963,'right_hand_thumb_0_joint':0.,
            'right_hand_thumb_1_joint':-1.0471976,'right_hand_thumb_2_joint':-1.5707963})


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        live = ObservationTermCfg(func=live_observation)
        def __post_init__(self):
            self.concatenate_terms=True
            self.enable_corruption=False
    policy: PolicyCfg=PolicyCfg()


@configclass
class EventsCfg:
    initialize=EventTermCfg(func=startup,mode='startup')
    reset=EventTermCfg(func=reset_scene,mode='reset')


@configclass
class RewardsCfg:
    alive=RewardTermCfg(func=mdp.is_alive,weight=1.)
    termination=RewardTermCfg(func=mdp.is_terminated,weight=-400.)
    joint_tracking=RewardTermCfg(func=tracking_error,weight=-.2,params={'kind':'joints'})
    position_tracking=RewardTermCfg(func=tracking_error,weight=-.2,params={'kind':'position'})
    orientation_tracking=RewardTermCfg(func=tracking_error,weight=-.2,params={'kind':'orientation'})
    keypoint_tracking=RewardTermCfg(func=tracking_error,weight=-.05,params={'kind':'keypoints'})
    hand_tracking=RewardTermCfg(func=tracking_error,weight=.3,params={'kind':'hand'})
    wrist_orientation=RewardTermCfg(func=tracking_error,weight=-1.,params={'kind':'wrist_orientation'})
    pick=RewardTermCfg(func=pick_success,weight=.5)
    action_rate=RewardTermCfg(func=mdp.action_rate_l2,weight=-.005)
    torques=RewardTermCfg(func=mdp.joint_torques_l2,weight=-1.5e-7)
    accelerations=RewardTermCfg(func=mdp.joint_acc_l2,weight=-1.25e-7)
    feet_slide=RewardTermCfg(func=foot_slide,weight=-.1)
    feet_parallel=RewardTermCfg(func=foot_tilt,weight=-1.)


@configclass
class TerminationsCfg:
    time_out=TerminationTermCfg(func=mdp.time_out,time_out=True)
    fall=TerminationTermCfg(func=fallen)
    reference_end=TerminationTermCfg(func=reference_finished,time_out=True)


@configclass
class Pick29EnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg=SceneCfg(num_envs=4,env_spacing=3.)
    observations: ObservationsCfg=ObservationsCfg()
    actions: ActionsCfg=ActionsCfg()
    events: EventsCfg=EventsCfg()
    rewards: RewardsCfg=RewardsCfg()
    terminations: TerminationsCfg=TerminationsCfg()
    commands=None
    curriculum=None
    reference_manifest: str=''
    def __post_init__(self):
        self.decimation=4
        self.episode_length_s=10.
        self.sim.dt=.005
        self.sim.render_interval=self.decimation
        self.sim.physx.gpu_max_rigid_contact_count=2**20
        self.sim.physx.gpu_max_rigid_patch_count=2**17

"""PhyGile observation/action contract on the official Unitree Mjlab G1 task.

Only frozen-policy validation is implemented here. It does not replace the
paper's MoEPPO training algorithm with the newer RSL-RL default.
"""
from dataclasses import dataclass,fields,replace
import copy,re
import numpy as np
import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationGroupCfg,ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking.mdp import MotionCommand,MotionCommandCfg
from mjlab.utils.lab_api.math import quat_mul,quat_inv,matrix_from_quat,quat_apply,quat_apply_inverse,yaw_quat
from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg


@dataclass(kw_only=True)
class PhyGileMotionCfg(MotionCommandCfg):
    future_steps:int=2
    encoder_future_steps:int=5
    encoder_future_stride:int=20
    def build(self,env):return PhyGileMotion(self,env)

class PhyGileMotion(MotionCommand):
    @property
    def command(self):
        offsets=[0]+list(range(1,self.cfg.future_steps+1))+[1+i*self.cfg.encoder_future_stride for i in range(self.cfg.encoder_future_steps)]
        pieces=[]
        for offset in offsets:
            ids=torch.clamp(self.time_steps+offset,max=self.motion.time_step_total-1)
            pieces.append(torch.cat([self.motion.joint_pos[ids][:,self.policy_joint_ids],self.motion.joint_vel[ids][:,self.policy_joint_ids],
              self.motion.body_pos_w[ids,self.motion_anchor_body_index]+self._env.scene.env_origins,self.motion.body_quat_w[ids,self.motion_anchor_body_index]],dim=-1))
        return torch.cat(pieces,dim=-1)
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.policy_joint_ids=torch.tensor([self.robot.joint_names.index(n) for n in env._phygile_contract['joint_names']],device=self.device)

def policy_observation(env):
    c=env._phygile_contract;robot=env.scene['robot'];mt=env.command_manager.get_term('motion')
    ids=mt.policy_joint_ids
    relative=quat_mul(quat_inv(mt.robot_anchor_quat_w),mt.anchor_quat_w)
    ori=matrix_from_quat(relative)[...,:2].reshape(env.num_envs,-1)
    defaults=torch.tensor(c['default_joint_pos'],device=env.device)
    return torch.cat([mt.command,ori,robot.data.root_link_ang_vel_b,robot.data.joint_pos[:,ids]-defaults,
      robot.data.joint_vel[:,ids],env.action_manager.action[:,ids]],dim=-1)

def timeout(env):return env.episode_length_buf>=env.max_episode_length
def bad_anchor_z(env):
    mt=env.command_manager.get_term('motion')
    return (mt.anchor_pos_w[:,2]-mt.robot_anchor_pos_w[:,2]).abs()>.25
def bad_orientation(env):
    mt=env.command_manager.get_term('motion');g=torch.tensor([0.,0.,-1.],device=env.device).expand(env.num_envs,-1)
    return (quat_apply_inverse(mt.anchor_quat_w,g)[:,2]-quat_apply_inverse(mt.robot_anchor_quat_w,g)[:,2]).abs()>.8
def bad_end_effectors(env):
    mt=env.command_manager.get_term('motion')
    indexes=[mt.cfg.body_names.index(n) for n in ['left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link']]
    return (mt.body_pos_relative_w[:,indexes,2]-mt.robot_body_pos_w[:,indexes,2]).abs().gt(.25).any(dim=1)

def initialize_relative(mt):
    n=len(mt.cfg.body_names)
    anchor=mt.anchor_pos_w[:,None,:].repeat(1,n,1)
    aq=mt.anchor_quat_w[:,None,:].repeat(1,n,1)
    rq=mt.robot_anchor_quat_w[:,None,:].repeat(1,n,1)
    pos=mt.robot_anchor_pos_w[:,None,:].repeat(1,n,1);pos[...,2]=anchor[...,2]
    rot=yaw_quat(quat_mul(rq,quat_inv(aq)))
    mt.body_quat_relative_w=quat_mul(rot,mt.body_quat_w)
    mt.body_pos_relative_w=pos+quat_apply(rot,mt.body_pos_w-anchor)

def convert_motion(source,destination,robot_joint_names,robot_body_names):
    with np.load(source,allow_pickle=False) as data:values={k:data[k] for k in data.files}
    joint_names=values['joint_names'].tolist();body_names=values['body_names'].tolist()
    joints=[joint_names.index(n) for n in robot_joint_names]
    bodies=[body_names.index(n) for n in robot_body_names]
    for key in ['joint_pos','joint_vel']:values[key]=values[key][:,joints]
    for key in ['body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']:values[key]=values[key][:,bodies]
    values['joint_names']=np.asarray(robot_joint_names);values['body_names']=np.asarray(robot_body_names)
    np.savez_compressed(destination,**values)
    return {'joint_mapping':joints,'body_mapping':bodies,'frames':len(values['joint_pos'])}

def make_config(contract,motion_file,frames,seed=2026,contact_profile='official'):
    cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False,play=True)
    cfg.seed=seed;cfg.scene.num_envs=1
    robot=cfg.scene.entities['robot'];names=contract['joint_names']
    robot.init_state.joint_pos=dict(zip(names,contract['default_joint_pos']))
    robot.articulation=EntityArticulationInfoCfg(actuators=tuple(BuiltinPositionActuatorCfg(target_names_expr=(re.escape(name),),
      stiffness=contract['stiffness'][i],damping=contract['damping'][i],effort_limit=contract['joint_effort_limits'][i],armature=contract['joint_armature'][i]) for i,name in enumerate(names)),soft_joint_pos_limit_factor=.9)
    robot.sort_actuators=True
    if contact_profile=='matched_sliding_friction':
        # Measured deterministic PhysX materials are all [1,1,0] on ground
        # [1,1,0], multiply combine. Match only the sliding coefficient.
        # Keep official geometry, condim, torsion/rolling and solver unchanged.
        material=np.asarray(contract['shape_material_properties'])
        ground=contract['ground_material_config']
        if not (np.allclose(material,[1.,1.,0.],atol=0,rtol=0) and
                ground['static_friction']==ground['dynamic_friction']==1. and
                ground['friction_combine_mode']=='multiply'):
            raise ValueError('This profile requires measured uniform source materials')
        robot.collisions=tuple(replace(x,friction={r'^(left|right)_foot[1-7]_collision$':(1.0,)}) for x in robot.collisions)
    elif contact_profile!='official':raise ValueError(contact_profile)
    action=cfg.actions['joint_pos']
    action.scale=dict(zip(names,contract['action_scale']))
    action.offset=dict(zip(names,contract['action_offset']))
    action.use_default_offset=False
    cfg.scene.sensors=()
    cfg.sim.mujoco.timestep=contract['physics_dt'];cfg.decimation=contract['decimation']
    cfg.episode_length_s=frames*contract['control_dt'];cfg.sim.nconmax=100;cfg.sim.njmax=1000
    original=cfg.commands['motion']
    cfg.commands={'motion':PhyGileMotionCfg(**{**{f.name:copy.deepcopy(getattr(original,f.name)) for f in fields(original)},'motion_file':str(motion_file),'joint_position_range':(0.,0.),'pose_range':{},'velocity_range':{},'sampling_mode':'start',
      'future_steps':contract['future_steps'],'encoder_future_steps':contract['encoder_future_steps'],'encoder_future_stride':contract['encoder_future_stride'],'debug_vis':False})}
    # The complete 616D policy observation is computed once in the original order.
    cfg.observations={'actor':ObservationGroupCfg(terms={'phygile':ObservationTermCfg(func=policy_observation)},concatenate_terms=True,enable_corruption=False)}
    cfg.rewards={};cfg.metrics={};cfg.events={}
    cfg.terminations={'time_out':TerminationTermCfg(func=timeout,time_out=True),'anchor_pos':TerminationTermCfg(func=bad_anchor_z),
      'anchor_ori':TerminationTermCfg(func=bad_orientation),'ee_body_pos':TerminationTermCfg(func=bad_end_effectors)}
    return cfg

class PhyGileMjlabEnv(ManagerBasedRlEnv):
    def __init__(self,cfg,contract,**kwargs):
        self._phygile_contract=contract
        super().__init__(cfg,**kwargs)

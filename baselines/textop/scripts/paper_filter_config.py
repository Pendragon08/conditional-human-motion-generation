"""Task-local paper overrides over pinned official TextOp Tracker classes.

AppendixA requires relaxed height termination DURING40k privileged pretraining.
No shared source edits. Unspecified EMA/repeat choices remain explicit.
"""
import pathlib
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from textop_tracker.tasks.tracking.tracking_env_cfg import PrivPrivObservationsCfg
from textop_tracker.tasks.tracking.config.g1.flat_env_cfg import G1FlatPrivPrivObsEnvCfg
from textop_tracker.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
import textop_tracker.tasks.tracking.mdp as mdp

@configclass
class PaperPrivileged(PrivPrivObservationsCfg.PrivilegedCfg):
    projected_gravity=ObsTerm(func=mdp.projected_gravity)

def offline_floor(path):
    """Local PhysX ground plane with exact paper1.0 friction, no remote texture."""
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema,PhysicsSchemaTools,Gf
    path=pathlib.Path(path)
    if path.exists():return str(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.CreateNew(str(path));root=UsdGeom.Xform.Define(stage,'/Ground');stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.0)
    PhysicsSchemaTools.addGroundPlane(stage,'/Ground/Plane','Z',2000.0,Gf.Vec3f(0,0,0),Gf.Vec3f(.5,.5,.5))
    material=UsdShade.Material.Define(stage,'/Ground/PhysicsMaterial');api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(1.0);api.CreateDynamicFrictionAttr(1.0);api.CreateRestitutionAttr(0.0)
    px=PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim());px.CreateFrictionCombineModeAttr('multiply');px.CreateRestitutionCombineModeAttr('multiply')
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,materialPurpose='physics')
    stage.GetRootLayer().Save();return str(path)

def make(spec,motion_files,usd_dir,phase='relaxed',num_envs=8192):
    assert phase in ['relaxed','strict']
    cfg=G1FlatPrivPrivObsEnvCfg();cfg.scene.num_envs=num_envs;cfg.seed=20261025;cfg.sim.device='cuda:1'
    cfg.scene.robot.spawn.asset_path=spec['robot_urdf'];cfg.scene.robot.spawn.usd_dir=str(pathlib.Path(usd_dir)/'robot')
    cfg.scene.robot.spawn.usd_file_name='g1.usd'
    cfg.scene.terrain.terrain_type='usd';cfg.scene.terrain.usd_path=offline_floor(pathlib.Path(usd_dir)/'floor.usda')
    cfg.scene.terrain.visual_material=None
    cfg.commands.motion.anchor_body_name='pelvis';cfg.commands.motion.motion_files=[str(x) for x in motion_files]
    cfg.commands.motion.future_steps=5;cfg.commands.motion.debug_vis=False;cfg.scene.contact_forces.debug_vis=False
    cfg.observations.policy=PaperPrivileged();cfg.observations.critic=PaperPrivileged()
    for group in [cfg.observations.policy,cfg.observations.critic]:
        group.enable_corruption=False;group.concatenate_terms=True
    for name in ['physics_material','add_joint_default_pos','base_com','push_robot']:setattr(cfg.events,name,None)
    cfg.rewards.feet_slide.weight=-.3;cfg.rewards.soft_landing.weight=-.0003
    cfg.rewards.overspeed.weight=-1.;cfg.rewards.overeffort.weight=-1.
    for name in ['feet_slide','soft_landing','overspeed','overeffort']:getattr(cfg.rewards,name).params['pfail_threshold']=1.0
    cfg.commands.motion.enable_adaptive_sampling=True
    # Defaults from the pinned release, not constants reported by the paper.
    cfg.commands.motion.ads_type='v1';cfg.commands.motion.adaptive_alpha=.001
    cfg.terminations.ee_body_pos.params['threshold']=.5
    if phase=='relaxed':cfg.terminations.anchor_pos=None;cfg.terminations.ee_body_pos=None
    agent=G1FlatPPORunnerCfg();agent.device='cuda:1';agent.seed=20261025
    agent.policy.actor_hidden_dims=[2048,1024,512];agent.policy.critic_hidden_dims=[2048,1024,512]
    agent.max_iterations=40000;agent.num_steps_per_env=24;agent.save_interval=5000;agent.logger='tensorboard'
    agent.experiment_name='textop_paper_privileged_filter'
    return cfg,agent

def paper_settings():
    return {'source':'https://arxiv.org/html/2602.07439v1#A1','filter_iterations':40000,'training_terminations':'relaxed height terms removed',
        'anchor':'pelvis','privileged_observation_dim_expected':557,'actor_hidden':[2048,1024,512],
        'critic_hidden':[2048,1024,512],'target_num_envs':8192,'steps_per_env':24,'learning_rate':.001,
        'minibatches':4,'epochs':5,'clip_param':.2,'simulation_hz':200,'policy_hz':50,
        'strict_pelvis_height_threshold_m':.25,'strict_end_effector_height_threshold_m':.5,
        'domain_randomization':False,'own_choices_not_specified_by_paper':{'adaptive_average':'pinned release EMA alpha0.001, ads_typev1',
        'strict_repetitions':None,'failure_window_or_chunking':None,'checkpoint_interval':5000,
        'asset':'shared BeyondMimic G1 description, authors exact asset hash unavailable',
        'floor':'locally generated PhysX ground plane with friction1.0; no remote visual textures'}}

"""New one-stage431/557 PPO configuration. Import only inside an authorized SDK job.

No teacher checkpoint is read. No model or SimulationApp is constructed here.
The caller must provide a real accepted, finite, mixed original/generated library.
"""
import pathlib
from user_step34_contract import validate


def make(policy,environment,mixed_receipt,motion_files,output,num_envs=None):
    validate(policy)
    assert policy['training_authorized'] is True or policy.get('witnesses_authorized') is True
    assert mixed_receipt['status']=='STEP3_REAL_AND_GENERATED_RL_LIBRARY_ACCEPTED'
    assert mixed_receipt['actual_generated_clips']==5368
    assert mixed_receipt['actual_generated_control_intervals']==5666400
    assert mixed_receipt['original_VAL_motions_in_training']==0 and mixed_receipt['numerical_failures_unaccepted']==0
    assert len(motion_files)==mixed_receipt['actual_mixed_clip_count'] and motion_files
    assert all(pathlib.Path(name).is_file() for name in motion_files)
    from textop_tracker.tasks.tracking.config.g1.flat_env_cfg import G1FlatProjGravObsEnvCfg
    from textop_tracker.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from textop_tracker.tasks.tracking.mdp.commands_multi import MotionCommand
    from textop_tracker.tasks.tracking import mdp
    from solid_floor_diagnostic import create
    cfg=G1FlatProjGravObsEnvCfg()
    cfg.seed=policy['seed'];cfg.scene.num_envs=policy['num_envs'] if num_envs is None else num_envs
    cfg.sim.device='cuda:1';cfg.sim.dt=.005;cfg.decimation=4;cfg.episode_length_s=10.
    cfg.scene.robot.spawn.asset_path=environment['robot_urdf']
    cfg.scene.robot.spawn.usd_dir=str(pathlib.Path(output)/'robot')
    cfg.scene.robot.spawn.usd_file_name='g1.usd'
    cfg.scene.terrain.terrain_type='usd';cfg.scene.terrain.usd_path=create(pathlib.Path(output)/'solid_floor.usda')
    cfg.scene.terrain.visual_material=None
    cfg.commands.motion.class_type=MotionCommand
    cfg.commands.motion.motion_files=[str(x) for x in motion_files]
    cfg.commands.motion.anchor_body_name='pelvis';cfg.commands.motion.future_steps=5
    cfg.commands.motion.debug_vis=False;cfg.scene.contact_forces.debug_vis=False
    cfg.observations.policy.enable_corruption=True
    cfg.observations.critic.enable_corruption=False
    assert cfg.observations.policy.projected_gravity is not None and cfg.observations.critic.projected_gravity is not None
    assert not hasattr(cfg.observations.policy,'body_pos') and not hasattr(cfg.observations.policy,'body_ori')
    assert cfg.observations.critic.body_pos is not None and cfg.observations.critic.body_ori is not None
    for group in [cfg.observations.policy,cfg.observations.critic]:group.concatenate_terms=True
    for name in ['physics_material','add_joint_default_pos','base_com','push_robot']:
        assert getattr(cfg.events,name) is not None, 'Paper domain randomization must remain enabled'
    cfg.terminations.anchor_pos.params['threshold']=.25
    cfg.terminations.anchor_ori.params['threshold']=.8
    cfg.terminations.ee_body_pos.params['threshold']=.5
    for name,weight,std in [('motion_global_anchor_pos',.5,.3),('motion_global_anchor_ori',.5,.4),
        ('motion_body_pos',1.,.3),('motion_body_ori',1.,.4),('motion_body_lin_vel',1.,1.),('motion_body_ang_vel',1.,3.14)]:
        term=getattr(cfg.rewards,name);term.weight=weight;term.params['std']=std
    for name,weight in [('action_rate_l2',-.1),('joint_limit',-10.),('undesired_contacts',-.1),
        ('feet_slide',-.3),('soft_landing',-.0003),('overspeed',-1.),('overeffort',-1.)]:
        getattr(cfg.rewards,name).weight=weight
    # TableVIII lists unconditional regularizers. Use the same released
    # numerical functions without the release's extra failure-rate mask.
    for name,function in [('feet_slide',mdp.feet_slide),('soft_landing',mdp.soft_landing),
        ('overspeed',mdp.joint_vel_out_of_manual_limit_reward),('overeffort',mdp.joint_effort_out_of_limit_fixed_reward)]:
        term=getattr(cfg.rewards,name);term.func=function;term.params.pop('pfail_threshold',None)
    cfg.commands.motion.enable_adaptive_sampling=True
    cfg.commands.motion.ads_type='v1';cfg.commands.motion.adaptive_alpha=.001
    agent=G1FlatPPORunnerCfg();agent.device='cuda:1';agent.seed=policy['seed']
    agent.policy.actor_hidden_dims=[2048,1024,512];agent.policy.critic_hidden_dims=[2048,1024,512]
    agent.policy.activation='elu';agent.num_steps_per_env=24
    agent.algorithm.learning_rate=.001;agent.algorithm.num_mini_batches=4
    agent.algorithm.num_learning_epochs=5;agent.algorithm.clip_param=.2
    agent.max_iterations=policy['target_RL_iterations'];agent.experiment_name='textop_step4_one_stage_augmentation'
    # Production wrapper must replace RSL's periodic saver with immutable accepted slots.
    agent.save_interval=policy['target_RL_iterations']+1
    return cfg,agent

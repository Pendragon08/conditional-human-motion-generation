"""Unactivated config preparation for new PPO5000 heldout physics.

This module is not a launcher, source acceptance gate or completed evaluator.
The caller must prove the fullstate/export, exact real FK inputs, source hashes,
exclusive model lock, resource budget and process closure before calling it.
"""
import pathlib


def make(request, environment, rows, output, assigned_motion_class, motion_complete):
    """Build a distinct nonprivileged 431D replay config inside an SDK job.

    Observation corruption and domain randomization are off in this diagnostic;
    this decision must be reported. This does not establish robustness under
    disturbances or count as a full online or physical-robot evaluation.
    """
    if request.get('authorized') is not True or request.get('stage') != 'heldout':
        raise ValueError('heldout request is not authorized')
    if request.get('observation_corruption') is not False or request.get('domain_randomization') is not False:
        raise ValueError('diagnostic conditions must be bound explicitly')
    if request.get('policy_hz') != 50 or request.get('future_reference_frames') != 5:
        raise ValueError('wrong policy timing')
    if request.get('num_envs') != 16 or type(request.get('seed')) is not int:
        raise ValueError('explicit 16-environment evaluation and seed required')
    from ppo5000_heldout_manifest_contract import validate as validate_manifest
    validate_manifest(rows, request.get('generation_seeds'))
    files = [r['file'] for r in rows]
    if any(not isinstance(x, str) or not pathlib.Path(x).is_absolute() for x in files):
        raise ValueError('invalid reference path')
    if not callable(motion_complete) or not isinstance(assigned_motion_class, type):
        raise ValueError('missing bound replay helpers')
    output = pathlib.Path(output)
    if not output.is_absolute() or output.exists():
        raise ValueError('new absolute private output required')
    # Imports and writes below are allowed only after the independent caller gate.
    from isaaclab.managers import TerminationTermCfg
    from textop_tracker.tasks.tracking.config.g1.flat_env_cfg import G1FlatProjGravObsEnvCfg
    from solid_floor_diagnostic import create
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    cfg = G1FlatProjGravObsEnvCfg()
    cfg.seed = request['seed']; cfg.scene.num_envs = request['num_envs']
    cfg.sim.device = 'cuda:1'; cfg.sim.dt = .005; cfg.decimation = 4
    cfg.scene.robot.spawn.asset_path = environment['robot_urdf']
    cfg.scene.robot.spawn.usd_dir = str(output / 'robot'); cfg.scene.robot.spawn.usd_file_name = 'g1.usd'
    cfg.scene.terrain.terrain_type = 'usd'; cfg.scene.terrain.usd_path = create(output / 'solid_floor.usda')
    cfg.scene.terrain.visual_material = None
    cfg.commands.motion.class_type = assigned_motion_class
    cfg.commands.motion.motion_files = files; cfg.commands.motion.anchor_body_name = 'pelvis'
    cfg.commands.motion.future_steps = 5; cfg.commands.motion.debug_vis = False
    cfg.scene.contact_forces.debug_vis = False
    cfg.observations.policy.enable_corruption = False
    cfg.observations.critic.enable_corruption = False
    for group in (cfg.observations.policy, cfg.observations.critic):
        group.concatenate_terms = True
    if hasattr(cfg.observations.policy, 'body_pos') or hasattr(cfg.observations.policy, 'body_ori'):
        raise ValueError('policy must not receive privileged body pose')
    if cfg.observations.policy.projected_gravity is None:
        raise ValueError('projected gravity is required')
    for name in ('physics_material', 'add_joint_default_pos', 'base_com', 'push_robot'):
        setattr(cfg.events, name, None)
    cfg.terminations.anchor_pos.params['threshold'] = .25
    cfg.terminations.anchor_ori.params['threshold'] = .8
    cfg.terminations.ee_body_pos.params['threshold'] = .5
    cfg.commands.motion.enable_adaptive_sampling = False
    cfg.commands.motion.start_from_zero_step = True
    cfg.commands.motion.pose_range = {}; cfg.commands.motion.velocity_range = {}
    cfg.commands.motion.joint_position_range = (0., 0.); cfg.commands.motion.random_static_prob = -1.
    cfg.episode_length_s = max(r['frames'] for r in rows) / 50. + .04
    cfg.terminations.time_out = None
    cfg.terminations.motion_complete = TerminationTermCfg(func=motion_complete, time_out=True)
    return cfg

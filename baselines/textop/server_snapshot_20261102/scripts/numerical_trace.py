"""Read-only, bounded telemetry around the unchanged full-library PPO rollout.

Contact sensor reads occur only after the existing reward manager consumed them.
Per-source/reset-state evidence stays private; the trace contains aggregates only.
Alert levels select diagnostic snapshots, never filter data or modify simulation.
"""
import hashlib
import json
import math
import pathlib
import torch


def install(wrapped, output, rows, spec):
    env = wrapped.unwrapped
    robot = env.scene['robot']
    motion = env.command_manager.get_term('motion')
    out = pathlib.Path(output)
    private = out / 'numerical_private'
    private.mkdir(exist_ok=False)
    result = {'physics_samples': 0, 'reward_samples': 0,
              'first_state_alert': None, 'first_reward_alert': None,
              'thresholds_are_diagnostic_only': True, 'simulation_or_reward_changes': False}
    trace = out / 'numerical_trace.jsonl'
    assert not trace.exists()

    def append(row):
        with trace.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')

    def stat(value):
        value = value.detach()
        finite = torch.isfinite(value)
        valid = value[finite]
        return {'min': float(valid.min()) if valid.numel() else None,
                'max': float(valid.max()) if valid.numel() else None,
                'abs_max': float(valid.abs().max()) if valid.numel() else None,
                'nonfinite': int((~finite).sum())}

    def snapshot(label, ids):
        ids = sorted(set(ids))[:spec['maximum_private_environments_per_event']]
        evidence = []
        for i in ids:
            idx = int(motion.motion_idx[i])
            evidence.append({'environment': i, 'motion_index': idx,
                             'source': rows[idx], 'reference_frame': int(motion.time_steps[i]),
                             'root_state_w': robot.data.root_state_w[i].tolist(),
                             'joint_pos': robot.data.joint_pos[i].tolist(),
                             'joint_vel': robot.data.joint_vel[i].tolist(),
                             'body_state_w': robot.data.body_state_w[i].tolist(),
                             'action': env.action_manager.action[i].tolist(),
                             'reference_joint_pos': motion.joint_pos[i].tolist(),
                             'reference_joint_vel': motion.joint_vel[i].tolist()})
        (private / (label + '.private.json')).write_text(json.dumps(evidence))

    def state(label):
        values = {'joint_velocity': robot.data.joint_vel,
                  'body_linear_velocity': robot.data.body_lin_vel_w,
                  'body_angular_velocity': robot.data.body_ang_vel_w,
                  'body_height': robot.data.body_pos_w[:, :, 2],
                  'root_height': robot.data.root_pos_w[:, 2],
                  'root_quaternion_norm': robot.data.root_quat_w.norm(dim=-1),
                  'joint_position': robot.data.joint_pos,
                  'applied_torque': robot.data.applied_torque,
                  'action': env.action_manager.action}
        record = {'kind': label, 'physics_sample': result['physics_samples'],
                  'reward_sample': result['reward_samples'],
                  'fields': {k: stat(v) for k, v in values.items()}}
        velocity = torch.stack([v.reshape(env.num_envs, -1).abs().amax(dim=1)
                                for k, v in values.items() if 'velocity' in k]).amax(dim=0)
        ids = (velocity > spec['state_velocity_alert']).nonzero().flatten().tolist()
        record['velocity_alert_environment_count'] = len(ids)
        append(record)
        if ids and result['first_state_alert'] is None:
            result['first_state_alert'] = record
            snapshot('first_state_alert', ids)

    # Initial state is saved before any policy step; it supports comparison with
    # the first extreme executed state without changing the RNG or reset policy.
    torch.save({'motion_idx': motion.motion_idx.detach().cpu(),
                'time_steps': motion.time_steps.detach().cpu(),
                'root_state_w': robot.data.root_state_w.detach().cpu(),
                'joint_pos': robot.data.joint_pos.detach().cpu(),
                'joint_vel': robot.data.joint_vel.detach().cpu()},
               private / 'initial_state.private.pth')
    state('initial')
    original_update = env.scene.update
    original_compute = env.reward_manager.compute

    def scene_update(dt, *args, **kwargs):
        value = original_update(dt, *args, **kwargs)
        if dt > 0:
            result['physics_samples'] += 1
            state('after_physics')
        return value

    def compute(dt, *args, **kwargs):
        value = original_compute(dt, *args, **kwargs)
        result['reward_samples'] += 1
        forces = env.scene['contact_forces'].data.net_forces_w
        record = {'kind': 'reward_before_reset', 'physics_sample': result['physics_samples'],
                  'reward_sample': result['reward_samples'], 'reward': stat(value),
                  'contact_force_components': stat(forces),
                  'weighted_reward_rates': {name: stat(env.reward_manager._step_reward[:, i])
                                           for i, name in enumerate(env.reward_manager._term_names)}}
        append(record)
        mask = ((value.abs() > spec['reward_alert']) |
                (forces.reshape(env.num_envs, -1).abs().amax(dim=1) > spec['contact_force_alert']))
        ids = mask.nonzero().flatten().tolist()
        if ids and result['first_reward_alert'] is None:
            result['first_reward_alert'] = record
            snapshot('first_reward_alert', ids)
        return value

    env.scene.update = scene_update
    env.reward_manager.compute = compute

    def finish():
        result.update(trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                      trace_bytes=trace.stat().st_size,
                      private_evidence_kept_on_server=True,
                      initial_state_is_not_checkpoint=True,
                      formal_filter_training_started=False, physical_filter_executed=False)
        (out / 'numerical_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        return result
    return finish

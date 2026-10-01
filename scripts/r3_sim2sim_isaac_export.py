"""Read-only official Isaac Lab observation/reference witness for frozen policies.

Exports runtime reference tensors rather than reconstructing FK with MuJoCo.
No learning, IK, reference edits, or policy updates are performed.
"""
import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--task', choices=['Kick', 'Punch'], required=True)
p.add_argument('--policy', required=True)
p.add_argument('--output', required=True)
AppLauncher.add_app_launcher_args(p)
args, unknown = p.parse_known_args()
sys.argv = [sys.argv[0]]
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import hashlib
import json
import numpy as np
import torch
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
task_id = f'Isaac-Motion-Tracking-{args.task}-v0'
cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
cfg.observations.policy.enable_corruption = False
cfg.seed = 0
env = gym.make(task_id, cfg=cfg)
env.reset(seed=0)
e = env.unwrapped
policy = torch.jit.load(args.policy, map_location=args.device).eval()
robot = e.scene['robot']
lib = e.motion_lib
arrays = {}
for name in ['transl', 'quats', 'dof_pos', 'global_keypts', 'local_keypts', 'grab_pos',
             'offsets', 'switch_idxs', '_motion_lengths', '_motion_num_frames', '_motion_dt']:
    arrays[name] = getattr(lib, name).detach().cpu().numpy()
np.savez_compressed(out / f'{args.task}_references.npz', **arrays)
meta = {'task': args.task, 'joint_names': robot.joint_names,
        'reference_joint_names': lib.joint_names, 'default_joint_pos': robot.data.default_joint_pos[0].cpu().tolist(),
        'default_joint_vel': robot.data.default_joint_vel[0].cpu().tolist(), 'step_dt': e.step_dt,
        'start_motion_times': e.start_motion_times.cpu().tolist(), 'reference_count': int(e.total_motions),
        'obs_terms': e.observation_manager.active_terms['policy'],
        'policy_sha256': hashlib.sha256(Path(args.policy).read_bytes()).hexdigest(),
        'observation_noise': False, 'num_actions': e.action_manager.total_action_dim,
        'reference_source': str(e.cfg.ref_motions_path), 'snapshots': []}
meta['actuators'] = {name: {
    'joint_names': actuator.joint_names,
    'stiffness': actuator.stiffness[0].cpu().tolist(),
    'damping': actuator.damping[0].cpu().tolist(),
    'effort_limit': actuator.effort_limit[0].cpu().tolist(),
    'armature': actuator.armature[0].cpu().tolist(),
} for name, actuator in robot.actuators.items()}
obs = e.observation_manager.compute()['policy']
for i in range(4):
    with torch.inference_mode():
        action = policy(obs)
    snapshot = {
        'obs': obs[0].cpu().numpy(), 'action': action[0].cpu().numpy(),
        'root_pos': (robot.data.root_pos_w - e.scene.env_origins)[0].cpu().numpy(),
        'root_quat': robot.data.root_quat_w[0].cpu().numpy(),
        'root_lin_vel_b': robot.data.root_lin_vel_b[0].cpu().numpy(),
        'root_ang_vel_b': robot.data.root_ang_vel_b[0].cpu().numpy(),
        'joint_pos': robot.data.joint_pos[0].cpu().numpy(),
        'joint_vel': robot.data.joint_vel[0].cpu().numpy(),
        'last_action': e.action_manager.action[0].cpu().numpy(),
        'motion_id': e.motion_ids[0].cpu().numpy(),
        'motion_time': (e.episode_length_buf * e.step_dt + e.start_motion_times)[0].cpu().numpy(),
    }
    np.savez_compressed(out / f'{args.task}_snapshot_{i}.npz', **snapshot)
    meta['snapshots'].append({'index': i, 'motion_id': int(e.motion_ids[0]),
                              'motion_time': float(snapshot['motion_time']), 'obs_dim': int(obs.shape[1])})
    obs, _, _, _, _ = env.step(action)
    obs = obs['policy']
(out / f'{args.task}_interface.json').write_text(json.dumps(meta, indent=2))
print('ISAAC_INTERFACE_WITNESS', args.task, obs.shape, int(e.total_motions), flush=True)
env.close()
app.close()

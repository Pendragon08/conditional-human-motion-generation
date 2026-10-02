"""One-environment read-only Isaac interface export for completed task policies.

No fitting, new RL, or mutation of the running official baseline queue.
"""
import argparse
from pathlib import Path
import sys
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--task', choices=['Jump', 'Sit', 'Pick', 'Bimanual-Pick'], required=True)
p.add_argument('--policy', required=True)
p.add_argument('--output', required=True)
AppLauncher.add_app_launcher_args(p)
a, _ = p.parse_known_args()
sys.argv = [sys.argv[0]]
launcher = AppLauncher(a)
app = launcher.app

import gymnasium as gym
import hashlib
import inspect
import json
import numpy as np
import torch
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
cfg = parse_env_cfg(f'Isaac-Motion-Tracking-{a.task}-v0', device=a.device, num_envs=1)
cfg.observations.policy.enable_corruption = False
cfg.seed = 0
cfg.sim.use_fabric = False
env = gym.make(f'Isaac-Motion-Tracking-{a.task}-v0', cfg=cfg)
env.reset(seed=0)
e = env.unwrapped
robot, lib = e.scene['robot'], e.motion_lib
policy = torch.jit.load(a.policy, map_location=a.device).eval()

arrays = {name: getattr(lib, name).detach().cpu().numpy() for name in
          ['transl', 'quats', 'dof_pos', 'global_keypts', 'local_keypts', 'grab_pos',
           'offsets', 'switch_idxs', '_motion_lengths', '_motion_num_frames', '_motion_dt']}
np.savez_compressed(out / f'{a.task}_references.npz', **arrays)
terms = []
for name, dims, c in zip(e.observation_manager.active_terms['policy'],
                      e.observation_manager._group_obs_term_dim['policy'],
                      e.observation_manager._group_obs_term_cfgs['policy']):
    terms.append({'name': name, 'dim': int(np.prod(dims)), 'function': c.func.__name__,
                  'source': inspect.getsourcefile(c.func), 'clip': c.clip,
                  'scale': c.scale, 'time_offset': c.params.get('time_offset', 0)})
meta = {'task': a.task, 'policy_path': a.policy,
        'policy_sha256': hashlib.sha256(Path(a.policy).read_bytes()).hexdigest(),
        'joint_names': robot.joint_names, 'body_names': robot.body_names,
        'reference_joint_names': lib.joint_names,
        'default_joint_pos': robot.data.default_joint_pos[0].cpu().tolist(),
        'default_joint_vel': robot.data.default_joint_vel[0].cpu().tolist(),
        'step_dt': e.step_dt, 'reference_count': int(e.total_motions),
        'obs_terms': terms, 'num_actions': e.action_manager.total_action_dim,
        'reference_source': str(e.cfg.ref_motions_path), 'observation_noise': False,
        'export_io_adaptation': 'official cfg.sim.use_fabric=False; no camera',
        'statics': [], 'snapshots': []}
meta['actuators'] = {name: {'joint_names': act.joint_names, **{
    k: getattr(act, k)[0].cpu().tolist() for k in
    ['stiffness', 'damping', 'effort_limit', 'armature']}} for name, act in robot.actuators.items()}
meta['hand_actions'] = {}
for name in ['left_hand_action', 'right_hand_action']:
    t = e.action_manager.get_term(name)
    meta['hand_actions'][name] = {'joint_names': t._joint_names,
        'open': t._open_command.cpu().tolist(), 'close': t._close_command.cpu().tolist()}
for name in ['seat', 'kitchen', 'platform']:
    c = getattr(e.cfg.scene, name, None)
    if c is not None and c.spawn is not None and hasattr(c.spawn, 'size'):
        meta['statics'].append({'name': name, 'size': list(c.spawn.size),
            'position': list(c.init_state.pos), 'quat': list(c.init_state.rot),
            'friction': float(c.spawn.physics_material.dynamic_friction)})
if a.task in ['Pick', 'Bimanual-Pick']:
    obj = e.scene['object']
    c = e.cfg.scene.object
    reset = e.event_manager.get_term_cfg('reset_object')
    meta['object'] = {'size': list(c.spawn.size), 'mass': float(obj.root_physx_view.get_masses()[0, 0]),
        'friction': float(c.spawn.physics_material.dynamic_friction),
        'reset_height': reset.params.get('height', .67), 'reset_offset': reset.params.get('offset', [0, 0]),
        'startup_mass_randomization': 'seed0 runtime actual mass; held fixed for every MuJoCo trial',
        'success_height': .95 if a.task == 'Pick' else .7}
meta['root_reset_offset_z'] = e.event_manager.get_term_cfg('reset_base').params.get('offset_z', .15)

obs = e.observation_manager.compute()['policy']
for i in range(30):
    with torch.inference_mode():
        action = policy(obs)
    snapshot = {'obs': obs[0].cpu().numpy(), 'action': action[0].cpu().numpy(),
        'root_pos': (robot.data.root_pos_w-e.scene.env_origins)[0].cpu().numpy(),
        'root_quat': robot.data.root_quat_w[0].cpu().numpy(),
        'root_lin_vel_b': robot.data.root_lin_vel_b[0].cpu().numpy(),
        'root_ang_vel_b': robot.data.root_ang_vel_b[0].cpu().numpy(),
        'joint_pos': robot.data.joint_pos[0].cpu().numpy(), 'joint_vel': robot.data.joint_vel[0].cpu().numpy(),
        'body_pos': (robot.data.body_pos_w-e.scene.env_origins.unsqueeze(1))[0].cpu().numpy(),
        'body_quat': robot.data.body_quat_w[0].cpu().numpy(),
        'last_action': e.action_manager.action[0].cpu().numpy(),
        'motion_id': e.motion_ids[0].cpu().numpy(),
        'motion_time': (e.episode_length_buf*e.step_dt+e.start_motion_times)[0].cpu().numpy()}
    np.savez_compressed(out / f'{a.task}_snapshot_{i}.npz', **snapshot)
    meta['snapshots'].append({'index': i, 'motion_id': int(e.motion_ids[0]),
        'motion_time': float(snapshot['motion_time']), 'obs_dim': int(obs.shape[1])})
    obs = env.step(action)[0]['policy']
(out / f'{a.task}_interface.json').write_text(json.dumps(meta, indent=2))
print('COMPLETED_TASK_ISAAC_EXPORT', a.task, tuple(obs.shape), int(e.total_motions), flush=True)
env.close()
app.close()

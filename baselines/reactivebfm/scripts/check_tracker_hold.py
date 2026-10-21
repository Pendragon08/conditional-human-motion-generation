"""CPU native-policy/MuJoCo static-reference diagnostic, independent of planner."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / 'upstream'), str(BASE / 'upstream/deploy'), str(Path(__file__).parent)]
p = argparse.ArgumentParser(); p.add_argument('--runtime-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True); p.add_argument('--steps', type=int, default=400)
a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
spec = importlib.util.spec_from_file_location('hold_online_runner', BASE / 'upstream/deploy/run_online_generation.py')
r = importlib.util.module_from_spec(spec); sys.modules[spec.name] = r; spec.loader.exec_module(r)
args = r._build_parser().parse_args(['--model_path', 'unused', '--policy_checkpoint',
    str(a.runtime_root / 'weights/ScaleBFM_m_model_22200.pt'), '--init_pose', 'A_pose',
    '--deploy_device', 'cpu', '--headless', '--no-reference_forcing',
    '--scalebridge_override', 'agent._target_=torch_bfm_agent.TorchBFMAgent'])
cfg = r._compose_config(args, [0, 1, 2, 3, 4, 5])
agent = r.instantiate(cfg.agent); meta = agent.get_meta_data()
env = r.instantiate(cfg.env, metadata_dict=meta)
pose = r._load_a_pose(args.a_pose_path)
env.reset(init_state_dict=r._motion36_to_init_state(pose, r.G1_29DOF_JOINT_NAMES, meta['joint_names']))
motion = np.repeat(pose[None], a.steps + 10, axis=0)
env.set_reference_motion({'root_pos': torch.from_numpy(motion[:, :3]),
    'root_quat_xyzw': torch.from_numpy(motion[:, 3:7]),
    'dof_pos_policy': torch.from_numpy(r._reorder_columns(motion[:, 7:], list(r.G1_29DOF_JOINT_NAMES), meta['joint_names']))})
states = []; start = time.monotonic(); sim_start = env.simulator.mujoco_data.time
with torch.inference_mode():
    for _ in range(a.steps):
        obs = env.refresh_observation(); target, action = agent.get_action(obs)
        env.apply_control(target, action)
        states.append(r._env_to_motion36(env, r.G1_29DOF_JOINT_NAMES))
states = np.stack(states)
np.savez_compressed(a.output / 'simulated_states.npz', qpos=states, frequency=1 / env.dt, quat_order='xyzw')
receipt = {'status': 'STATIC_REFERENCE_PHYSICS_EXECUTED', 'control_steps': len(states),
    'simulation_seconds': float(env.simulator.mujoco_data.time - sim_start),
    'elapsed_seconds': time.monotonic() - start, 'finite': bool(np.isfinite(states).all()),
    'minimum_root_height_m': float(states[:, 2].min()), 'final_root_height_m': float(states[-1, 2]),
    'upright_entire_rollout_height_proxy': bool(states[:, 2].min() >= .4),
    'root_position_rmse_m': float(np.sqrt(np.mean((states[:, :3] - pose[:3]) ** 2))),
    'dof_rmse_rad': float(np.sqrt(np.mean((states[:, 7:] - pose[7:]) ** 2))),
    'reference_forcing': False, 'planner_used': False,
    'scope': 'Static A-pose native tracker diagnostic, not a generative-planning or paper metric.'}
env.simulator.close(); (a.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)

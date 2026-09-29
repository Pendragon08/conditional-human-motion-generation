"""Inference-only physics probe of the sampled training-state viewer pathway.

This is not PPO training or a result metric. It reuses a fixed official Kick
checkpoint to produce moving Isaac Lab states and sends one environment's pose
through the exact optional telemetry wrapper planned for matched training.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Motion-Tracking-Kick-v0")
parser.add_argument("--frames", type=int, default=1000)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--telemetry-udp", default="127.0.0.1:40123")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from r3_i1_training_telemetry import SampledTrainingTelemetry  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    raw = gym.make(args.task, cfg=env_cfg)
    raw = SampledTrainingTelemetry(raw, args.telemetry_udp)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs, _ = env.get_observations()
    for frame in range(args.frames):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs).clone())
        if frame % 100 == 0:
            print(f"R3_TELEMETRY_PROBE frame={frame} no_training=true", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

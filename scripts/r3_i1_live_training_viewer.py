"""Independent Isaac Sim view of sampled *actual* PPO simulation poses.

The viewer receives best-effort UDP packets from r3_i1_train_matched.py on
server loopback. It never connects back to the trainer; disconnecting WebRTC
or killing this process cannot stop PPO. This is a sampled reconstruction of
one training environment, not every frame of all parallel environments.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--telemetry-port", type=int, default=40123)
parser.add_argument("--status-path", type=str, required=True)
parser.add_argument("--timeout-seconds", type=int, default=0)
parser.add_argument("--self-test", action="store_true",
                    help="Move a synthetic pose to test the renderer; never label as training.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg("Isaac-Motion-Tracking-Kick-v0", device=args.device, num_envs=1)
    cfg.viewer.eye = (2.5, -5.0, 4.0)
    cfg.viewer.lookat = (2.0, 0.0, 0.7)
    env = gym.make("Isaac-Motion-Tracking-Kick-v0", cfg=cfg)
    raw = env.unwrapped
    robot = raw.scene["robot"]
    try:
        obj = raw.scene["object"]
    except KeyError:
        obj = None
    names = list(robot.data.joint_names)
    device = raw.device
    status = Path(args.status_path)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", args.telemetry_port))
    udp.setblocking(False)
    status.write_text("READY mode=sampled_actual_training_pose waiting_for_telemetry\n")
    start = time.monotonic()
    last_step = -1
    last_at = 0.0
    self_test_step = 0
    try:
        while simulation_app.is_running():
            if args.timeout_seconds and time.monotonic() - start > args.timeout_seconds:
                break
            newest = None
            while True:
                try:
                    packet, _ = udp.recvfrom(65535)
                    newest = json.loads(packet)
                except BlockingIOError:
                    break
                except (ValueError, OSError):
                    break
            if newest is None and args.self_test:
                self_test_step += 1
                root = robot.data.default_root_state[0, :7].detach().cpu().float().tolist()
                root[2] += 0.08 * math.sin(self_test_step / 20)
                q = robot.data.default_joint_pos[0].detach().cpu().float().tolist()
                if q:
                    q[0] += 0.18 * math.sin(self_test_step / 15)
                newest = {
                    "source": "viewer_self_test",
                    "robot_root_pose_wxyz": root,
                    "joint_names": names,
                    "joint_position_rad": q,
                    "scene_object_position_xyz_m": [2.1, 0.0, 0.67],
                    "env_step": self_test_step,
                }
            if newest is not None:
                try:
                    if newest["source"] != ("viewer_self_test" if args.self_test else "actual_training_simulation_sample"):
                        raise ValueError("Unexpected telemetry source")
                    if list(newest["joint_names"]) != names:
                        raise ValueError("Joint order mismatch")
                    root = torch.tensor([newest["robot_root_pose_wxyz"]], device=device)
                    q = torch.tensor([newest["joint_position_rad"]], device=device)
                    robot.write_root_pose_to_sim(root)
                    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
                    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
                    xyz = newest.get("scene_object_position_xyz_m")
                    if obj is not None and xyz is not None:
                        obj_pose = torch.tensor([[*xyz, 1.0, 0.0, 0.0, 0.0]], device=device)
                        obj.write_root_pose_to_sim(obj_pose)
                        obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
                    last_step = int(newest["env_step"])
                    last_at = time.time()
                    if last_step == 1:
                        print(f"R3_VIEWER_APPLIED_POSE mode={newest['source']}", flush=True)
                    mode = "synthetic_self_test_not_training" if args.self_test else "sampled_actual_training_pose"
                    status.write_text(
                        f"LIVE mode={mode} step={last_step} "
                        f"packet_age_s=0 training_unaffected=true\n"
                    )
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    status.write_text(f"TELEMETRY_ERROR {type(exc).__name__}: {exc}\n")
            elif last_step >= 0 and time.time() - last_at > 30:
                status.write_text(
                    f"STALE mode=sampled_actual_training_pose last_step={last_step} "
                    f"age_s={time.time()-last_at:.1f} training_unaffected=true\n"
                )
            raw.sim.render()
            simulation_app.update()
            time.sleep(0.05)
    finally:
        udp.close()
        env.close()
        simulation_app.close()
        status.write_text("STOPPED training_unaffected=true\n")


if __name__ == "__main__":
    main()

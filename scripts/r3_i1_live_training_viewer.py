"""Independent Isaac Sim view of sampled *actual* PPO simulation poses.

The viewer receives best-effort UDP packets from r3_i1_train_matched.py on
server loopback. It never connects back to the trainer; disconnecting WebRTC
or killing this process cannot stop PPO. This is a sampled reconstruction of
one training environment, not every frame of all parallel environments.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import socket
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Motion-Tracking-Kick-v0")
parser.add_argument("--telemetry-port", type=int, default=40123)
parser.add_argument("--status-path", type=str, required=True)
parser.add_argument("--timeout-seconds", type=int, default=0)
parser.add_argument("--self-test", action="store_true",
                    help="Move a synthetic pose to test the renderer; never label as training.")
parser.add_argument("--snapshot-path", default="",
                    help="Optional viewport capture for independent video diagnostics.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.viewer.eye = (3.0, -4.0, 2.4)
    cfg.viewer.lookat = (0.0, 0.0, 0.8)
    env = gym.make(args.task, cfg=cfg)
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
    capture_task = None
    next_capture = 0.0
    last_packet = None

    async def publish_snapshot(packet):
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

        target = Path(args.snapshot_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name("next-frame.png")
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("No active viewport for capture")
        capture = capture_viewport_to_file(viewport, str(temporary))
        await capture.wait_for_result()
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("Viewport capture produced no image")
        temporary.replace(target)
        metadata = {
            "captured_at_unix": time.time(),
            "source": packet["source"],
            "sampled_env_step": packet["env_step"],
            "telemetry_wall_time": packet.get("wall_time"),
            "robot_root_pose_wxyz": packet["robot_root_pose_wxyz"],
            "joint_position_rad": packet["joint_position_rad"],
            "description": "Rendered Isaac Sim viewport of actual training state sample; not every training frame.",
        }
        temp_meta = target.with_suffix(".next.json")
        temp_meta.write_text(json.dumps(metadata) + "\n")
        temp_meta.replace(target.with_suffix(".json"))
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
                    last_packet = newest
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
            # Flush the received articulation state before Hydra rendering.
            # Do not advance the independent world's physics: a second generic
            # app.update() can run its default actuators and replace the
            # sampled training pose with the viewer's idle pose.
            raw.sim.forward()
            raw.sim.set_setting("/app/player/playSimulations", False)
            simulation_app.update()
            raw.sim.set_setting("/app/player/playSimulations", True)
            if args.snapshot_path and last_packet is not None and time.monotonic() >= next_capture:
                if capture_task is None or capture_task.done():
                    if capture_task is not None:
                        try:
                            capture_task.result()
                        except Exception as exc:
                            print(f"R3_VIEWPORT_CAPTURE_ERROR {type(exc).__name__}: {exc}", flush=True)
                    capture_task = asyncio.ensure_future(publish_snapshot(dict(last_packet)))
                    next_capture = time.monotonic() + 1.0
            time.sleep(0.05)
    finally:
        udp.close()
        env.close()
        simulation_app.close()
        status.write_text("STOPPED training_unaffected=true\n")


if __name__ == "__main__":
    main()

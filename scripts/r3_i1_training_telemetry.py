"""Publish sampled *actual training simulation* poses without a viewer dependency.

One environment is sampled every 5 simulation steps. UDP is non-blocking and
best effort: a missing, slow or crashed viewer cannot stall PPO. No reward,
action, reset, state or optimizer value is changed by this wrapper.
"""

from __future__ import annotations

import json
import socket
import time

import gymnasium as gym


class SampledTrainingTelemetry(gym.Wrapper):
    def __init__(self, env: gym.Env, endpoint: str, stride: int = 5):
        super().__init__(env)
        host, port = endpoint.rsplit(":", 1)
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("Training telemetry must stay on server loopback")
        self._destination = ("127.0.0.1", int(port))
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._stride = stride
        self._steps = 0
        self._warned = False

    def step(self, action):
        result = self.env.step(action)
        self._steps += 1
        if self._steps % self._stride == 0:
            self._emit()
        return result

    def _emit(self) -> None:
        try:
            raw = self.env.unwrapped
            robot = raw.scene["robot"]
            origin = raw.scene.env_origins[0]
            root_pose = (
                robot.data.root_pos_w[0].detach().cpu().float().tolist()
                + robot.data.root_quat_w[0].detach().cpu().float().tolist()
            )
            root_pose[:3] = [root_pose[i] - float(origin[i]) for i in range(3)]
            joints = robot.data.joint_pos[0].detach().cpu().float().tolist()
            # Kick has a virtual reward target and no physical object in its
            # Isaac Lab scene. Other tasks may contain a rendered object.
            obj = None
            try:
                scene_object = raw.scene["object"]
            except KeyError:
                scene_object = None
            if scene_object is not None:
                position = scene_object.data.root_pos_w[0].detach().cpu().float().tolist()
                obj = [position[i] - float(origin[i]) for i in range(3)]
            packet = {
                "source": "actual_training_simulation_sample",
                "schema": 1,
                "wall_time": time.time(),
                "env_step": self._steps,
                "robot_root_pose_wxyz": root_pose,
                "joint_names": list(robot.data.joint_names),
                "joint_position_rad": joints,
                "scene_object_position_xyz_m": obj,
            }
            self._socket.sendto(
                json.dumps(packet, separators=(",", ":")).encode("utf-8"),
                self._destination,
            )
        except Exception as exc:
            # Monitoring is strictly optional; a crashed viewer or malformed
            # telemetry packet must not change the scientific training run.
            if not self._warned:
                print(f"R3_TELEMETRY_WARNING {type(exc).__name__}: {exc}", flush=True)
                self._warned = True

    def close(self):
        try:
            self._socket.close()
        finally:
            return self.env.close()

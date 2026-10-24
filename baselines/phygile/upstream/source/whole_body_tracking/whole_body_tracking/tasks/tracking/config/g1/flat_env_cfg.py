from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg

_CFG_ENV_VAR = "G1_TRAINING_CFG_PATH"
_DEFAULT_CFG_PATH = Path(__file__).resolve().parent / "cfg" / "g1_training.yaml"


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"G1 config '{path}' must contain a mapping at the top level.")
    return data


_CFG_PATH = Path(os.environ.get(_CFG_ENV_VAR, str(_DEFAULT_CFG_PATH))).expanduser()
_CFG_DICT = _load_yaml_config(_CFG_PATH)
_COMMAND_OVERRIDES: dict[str, Any] = _CFG_DICT.get("commands", {}) or {}
_MOTION_OVERRIDES: dict[str, Any] = _COMMAND_OVERRIDES.get("motion", {}) or {}


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
        self._apply_motion_overrides()

    def _apply_motion_overrides(self):
        if not _MOTION_OVERRIDES:
            return
        motion_cfg = self.commands.motion
        for key, value in _MOTION_OVERRIDES.items():
            if hasattr(motion_cfg, key):
                setattr(motion_cfg, key, value)
            else:
                print(f"[G1FlatEnvCfg] Unknown motion override key '{key}', skipping.")


@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE

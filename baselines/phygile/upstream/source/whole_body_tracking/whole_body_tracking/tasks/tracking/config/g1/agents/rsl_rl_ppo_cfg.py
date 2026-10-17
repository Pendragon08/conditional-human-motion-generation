from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

_CFG_ENV_VAR = "G1_TRAINING_CFG_PATH"
_DEFAULT_CFG_PATH = Path(__file__).resolve().parent.parent / "cfg" / "g1_training.yaml"


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"G1 config file '{path}' must contain a mapping at the top level.")
    return data


def _merge_defaults(defaults: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(defaults)
    if overrides:
        merged.update(overrides)
    return merged


_CFG_PATH = Path(os.environ.get(_CFG_ENV_VAR, str(_DEFAULT_CFG_PATH))).expanduser()
_CFG_DICT = _load_yaml_config(_CFG_PATH)
_RUNNER_CFG = _merge_defaults(
    {
        "num_steps_per_env": 24,
        "max_iterations": 30000,
        "save_interval": 500,
        "experiment_name": "g1_flat",
        "empirical_normalization": True,
        "upload_models_to_wandb": True,
    },
    _CFG_DICT.get("runner"),
)
_POLICY_CFG = _merge_defaults(
    {
        "class_name": "EncoderActorCritic",
        "init_noise_std": 1.0,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "encoder_hidden_dims": None,
        "share_encoder": None,
    },
    _CFG_DICT.get("policy"),
)
_ALGORITHM_CFG = _merge_defaults(
    {
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.005,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.013,  # 0.01->0.013增加此值可减慢学习率衰减速度
        "max_grad_norm": 1.0,
    },
    _CFG_DICT.get("algorithm"),
)


@configclass
class G1ActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-Critic 扩展，允许（可选）额外的 encoder MLP。"""

    class_name: str = _POLICY_CFG.get("class_name", "EncoderActorCritic")
    encoder_hidden_dims: list[int] | None = None
    share_encoder: bool | None = None
    future_command_encoder: dict[str, Any] | None = None
    # MoE 配置（仅当 class_name == "MoEActorCritic" 时生效）
    moe_config: dict[str, Any] | None = None


@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = _RUNNER_CFG["num_steps_per_env"]
    max_iterations = _RUNNER_CFG["max_iterations"]
    save_interval = _RUNNER_CFG["save_interval"]
    experiment_name = _RUNNER_CFG["experiment_name"]
    empirical_normalization = _RUNNER_CFG["empirical_normalization"]
    upload_models_to_wandb = _RUNNER_CFG.get("upload_models_to_wandb", True)
    policy = G1ActorCriticCfg(**_POLICY_CFG)
    algorithm = RslRlPpoAlgorithmCfg(**_ALGORITHM_CFG)


LOW_FREQ_SCALE = 0.5


@configclass
class G1FlatLowFreqPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal


def _activation_factory(name: str) -> Callable[[], nn.Module]:
    name = name.lower()
    mapping = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    return mapping.get(name, nn.ELU)


class _MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], activation_ctor: Callable[[], nn.Module]):
        super().__init__()
        hidden_dims = list(hidden_dims or [])
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_ctor())
            prev_dim = hidden_dim
        if layers:
            self.encoder = nn.Sequential(*layers)
        else:
            self.encoder = nn.Identity()
        self.output_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation_ctor: Callable[[], nn.Module],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation_ctor())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


@dataclass
class _FutureCommandEncoderCfg:
    per_frame_dim: int
    immediate_steps: int
    conv_steps: int
    conv_channels: Sequence[int]
    kernel_size: int
    dropout: float
    pooling: str
    latent_dim: int | None
    share_across_actor_critic: bool
    apply_to_actor: bool
    apply_to_critic: bool


class _FutureCommandProcessor(nn.Module):
    """Optionally processes the leading command segment via temporal convolutions."""

    def __init__(self, cfg: _FutureCommandEncoderCfg, activation_ctor: Callable[[], nn.Module]):
        super().__init__()
        self.per_frame_dim = max(int(cfg.per_frame_dim), 1)
        self.immediate_steps = max(int(cfg.immediate_steps), 0)
        self.conv_steps = max(int(cfg.conv_steps), 0)
        self.raw_frame_count = 1 + self.immediate_steps
        self.raw_dim = self.per_frame_dim * self.raw_frame_count
        self.input_dim = self.raw_dim + self.per_frame_dim * self.conv_steps
        self.pool: nn.Module = nn.Identity()

        conv_channels = list(cfg.conv_channels or [])
        latent_dim = None if cfg.latent_dim is None else int(cfg.latent_dim)

        if self.conv_steps == 0 or not conv_channels:
            self.conv = None
            self.latent_proj = None
            self.output_dim = self.raw_dim
            return

        kernel_size = max(int(cfg.kernel_size), 1)
        padding = kernel_size // 2
        prev_channels = self.per_frame_dim
        layers: list[nn.Module] = []
        dropout_p = max(float(cfg.dropout), 0.0)
        for channel in conv_channels:
            channel = int(channel)
            if channel <= 0:
                raise ValueError("conv_channels entries must be > 0.")
            layers.append(nn.Conv1d(prev_channels, channel, kernel_size, padding=padding))
            layers.append(activation_ctor())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            prev_channels = channel
        self.conv = nn.Sequential(*layers)

        pooling = cfg.pooling.lower()
        if pooling in ("avg", "mean"):
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError(f"Unsupported pooling '{cfg.pooling}', expected 'avg'/'mean' or 'max'.")

        if latent_dim is not None and latent_dim > 0:
            self.latent_proj = nn.Linear(prev_channels, latent_dim)
            latent_channels = latent_dim
        else:
            self.latent_proj = None
            latent_channels = prev_channels
        self.output_dim = self.raw_dim + latent_channels

    def forward(self, command_obs: torch.Tensor) -> torch.Tensor:
        if command_obs.shape[-1] < self.input_dim:
            raise RuntimeError(
                f"FutureCommandProcessor expected at least {self.input_dim} dims, "
                f"got {command_obs.shape[-1]}"
            )
        raw = command_obs[:, : self.raw_dim]
        if self.conv is None or self.conv_steps == 0:
            return raw
        conv_chunk = command_obs[
            :, self.raw_dim : self.raw_dim + self.per_frame_dim * self.conv_steps
        ]
        future = conv_chunk.view(conv_chunk.shape[0], self.conv_steps, self.per_frame_dim)
        future = future.transpose(1, 2)  # (B, per_frame_dim, conv_steps)
        latent = self.conv(future)
        latent = self.pool(latent).squeeze(-1)
        if self.latent_proj is not None:
            latent = self.latent_proj(latent)
        return torch.cat([raw, latent], dim=-1)

    def forward_with_latent(self, command_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Like ``forward()`` but also returns the conv latent (*z_t*) separately.

        Returns:
            processed: Same as ``forward()`` output — ``cat([raw, latent])``.
            z_t: The conv latent vector ``(B, latent_dim)``.  Zero-length
                 last dim if no conv is configured.
        """
        raw = command_obs[:, : self.raw_dim]
        if self.conv is None or self.conv_steps == 0:
            return raw, torch.zeros(raw.shape[0], 0, device=raw.device)
        conv_chunk = command_obs[
            :, self.raw_dim : self.raw_dim + self.per_frame_dim * self.conv_steps
        ]
        future = conv_chunk.view(conv_chunk.shape[0], self.conv_steps, self.per_frame_dim)
        future = future.transpose(1, 2)  # (B, per_frame_dim, conv_steps)
        latent = self.conv(future)
        latent = self.pool(latent).squeeze(-1)
        if self.latent_proj is not None:
            latent = self.latent_proj(latent)
        return torch.cat([raw, latent], dim=-1), latent

    @property
    def latent_output_dim(self) -> int:
        """Dimensionality of the conv latent (z_t).  0 when no conv is configured."""
        if self.conv is None:
            return 0
        return self.output_dim - self.raw_dim


class _EncoderActorCriticBase(nn.Module):
    """Shared encoder / FutureCommandProcessor / critic / noise logic.

    Both ``EncoderActorCritic`` (MLP actor) and ``MoEActorCritic``
    (MoE actor) inherit from this class so the shared plumbing is
    written only once.
    """

    is_recurrent = False

    def _init_base(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: Sequence[int],
        critic_hidden_dims: Sequence[int],
        activation: str,
        init_noise_std: float,
        noise_std_type: str,
        encoder_hidden_dims: Sequence[int] | None,
        share_encoder: bool | None,
        future_command_encoder: dict | None,
    ) -> int:
        """Shared initialisation.  Returns ``actor_encoded_dim``."""
        self.num_actions = int(num_actions)
        self._actor_obs_dim = int(num_actor_obs)
        self._critic_obs_dim = int(num_critic_obs)
        self._activation_ctor = _activation_factory(activation)

        encoder_hidden_dims = list(encoder_hidden_dims or [])
        if share_encoder is None:
            share_encoder = self._actor_obs_dim == self._critic_obs_dim
        self._share_encoder = bool(share_encoder)

        self._actor_command_processor: _FutureCommandProcessor | None = None
        self._critic_command_processor: _FutureCommandProcessor | None = None
        self._actor_obs_processed_dim = self._actor_obs_dim
        self._critic_obs_processed_dim = self._critic_obs_dim

        if future_command_encoder:
            cfg = self._build_future_command_cfg(future_command_encoder)
            if cfg.apply_to_actor:
                self._actor_command_processor = _FutureCommandProcessor(cfg, self._activation_ctor)
                self._actor_obs_processed_dim = self._compute_processed_dim(
                    self._actor_obs_dim, self._actor_command_processor
                )
            if cfg.apply_to_critic:
                if cfg.share_across_actor_critic and cfg.apply_to_actor:
                    self._critic_command_processor = self._actor_command_processor
                else:
                    self._critic_command_processor = _FutureCommandProcessor(cfg, self._activation_ctor)
                self._critic_obs_processed_dim = self._compute_processed_dim(
                    self._critic_obs_dim, self._critic_command_processor
                )

        # ---- encoders ----
        if self._share_encoder:
            target_dim = max(self._actor_obs_processed_dim, self._critic_obs_processed_dim)
            self._shared_encoder = _MLPEncoder(target_dim, encoder_hidden_dims, self._activation_ctor)
            self._actor_adapter = (
                nn.Identity()
                if self._actor_obs_processed_dim == target_dim
                else nn.Linear(self._actor_obs_processed_dim, target_dim)
            )
            self._critic_adapter = (
                nn.Identity()
                if self._critic_obs_processed_dim == target_dim
                else nn.Linear(self._critic_obs_processed_dim, target_dim)
            )
            actor_encoded_dim = self._shared_encoder.output_dim
            critic_encoded_dim = self._shared_encoder.output_dim
        else:
            if encoder_hidden_dims:
                self._actor_encoder = _MLPEncoder(
                    self._actor_obs_processed_dim, encoder_hidden_dims, self._activation_ctor
                )
                self._critic_encoder = _MLPEncoder(
                    self._critic_obs_processed_dim, encoder_hidden_dims, self._activation_ctor
                )
            else:
                self._actor_encoder = nn.Identity()
                self._critic_encoder = nn.Identity()
                self._actor_encoder.output_dim = self._actor_obs_processed_dim  # type: ignore[attr-defined]
                self._critic_encoder.output_dim = self._critic_obs_processed_dim  # type: ignore[attr-defined]
            actor_encoded_dim = getattr(self._actor_encoder, "output_dim", self._actor_obs_processed_dim)
            critic_encoded_dim = getattr(self._critic_encoder, "output_dim", self._critic_obs_processed_dim)

        # ---- critic head (shared between MLP-actor and MoE-actor) ----
        self.critic = _build_mlp(critic_encoded_dim, critic_hidden_dims, 1, self._activation_ctor)

        # ---- action noise ----
        init_noise_value = float(init_noise_std)
        self._noise_std_type = noise_std_type
        if self._noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_value * torch.ones(self.num_actions))
        elif self._noise_std_type == "log":
            self.log_std = nn.Parameter(torch.full((self.num_actions,), math.log(init_noise_value)))
        else:
            raise ValueError(f"Unknown noise_std_type '{noise_std_type}', expected 'scalar' or 'log'.")
        self.distribution = None
        Normal.set_default_validate_args(False)

        return actor_encoded_dim

    # ---- encoding helpers ----
    def _encode_actor_obs(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self._apply_command_processor(obs, self._actor_command_processor)
        if self._share_encoder:
            scaled = self._actor_adapter(obs)
            return self._shared_encoder(scaled)
        return self._actor_encoder(obs)

    def _encode_critic_obs(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self._apply_command_processor(obs, self._critic_command_processor)
        if self._share_encoder:
            scaled = self._critic_adapter(obs)
            return self._shared_encoder(scaled)
        return self._critic_encoder(obs)

    def _current_std(self) -> torch.Tensor:
        if self._noise_std_type == "scalar":
            std = self.std.clamp(min=1e-6)
            if not torch.isfinite(std).all():
                std = std.nan_to_num(nan=0.3, posinf=1.0, neginf=1e-6)
            return std
        return self.log_std.exp().clamp(min=1e-6)

    # ---- shared API used by PPO ----
    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution is not initialized.")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        encoded = self._encode_critic_obs(critic_observations)
        return self.critic(encoded)

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution is not initialized.")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self._current_std()

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution is not initialized.")
        ent = self.distribution.entropy()
        return ent.sum(dim=-1) if ent.dim() > 1 else ent

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    # ---- internal helpers ----
    @staticmethod
    def _apply_command_processor(
        obs: torch.Tensor, processor: _FutureCommandProcessor | None
    ) -> torch.Tensor:
        if processor is None:
            return obs
        cmd_dim = processor.input_dim
        if obs.shape[-1] < cmd_dim:
            raise RuntimeError(
                f"Observation dim {obs.shape[-1]} is smaller than required command dim {cmd_dim}."
            )
        command = obs[:, :cmd_dim]
        remainder = obs[:, cmd_dim:]
        processed = processor(command)
        if remainder.numel() == 0:
            return processed
        return torch.cat([processed, remainder], dim=-1)

    @staticmethod
    def _compute_processed_dim(
        raw_dim: int, processor: _FutureCommandProcessor | None
    ) -> int:
        if processor is None:
            return raw_dim
        if raw_dim < processor.input_dim:
            raise ValueError(
                f"Observation dim ({raw_dim}) is smaller than command dim ({processor.input_dim})."
            )
        return raw_dim - processor.input_dim + processor.output_dim

    def _build_future_command_cfg(self, raw_cfg: dict) -> _FutureCommandEncoderCfg:
        def _int_or_default(value, default):
            if value is None:
                return int(default)
            return int(value)

        per_frame_dim_default = 2 * self.num_actions + 7
        per_frame_dim = _int_or_default(raw_cfg.get("per_frame_dim"), per_frame_dim_default)
        immediate_steps = _int_or_default(raw_cfg.get("immediate_steps"), 0)
        conv_steps = raw_cfg.get("conv_steps")
        if conv_steps is None:
            conv_steps = raw_cfg.get("future_steps", 0)
        conv_steps = _int_or_default(conv_steps, 0)
        conv_channels = raw_cfg.get("conv_channels", (256, 256))
        kernel_size = _int_or_default(raw_cfg.get("kernel_size"), 3)
        dropout = float(raw_cfg.get("dropout", 0.0))
        pooling = raw_cfg.get("pooling", "avg")
        latent_dim_value = raw_cfg.get("latent_dim", 128)
        latent_dim = None if latent_dim_value is None else _int_or_default(latent_dim_value, 0)
        share = bool(raw_cfg.get("share_across_actor_critic", True))
        apply_actor = bool(raw_cfg.get("apply_to_actor", True))
        apply_critic = bool(raw_cfg.get("apply_to_critic", True))
        return _FutureCommandEncoderCfg(
            per_frame_dim=per_frame_dim,
            immediate_steps=immediate_steps,
            conv_steps=conv_steps,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            pooling=pooling,
            latent_dim=latent_dim,
            share_across_actor_critic=share,
            apply_to_actor=apply_actor,
            apply_to_critic=apply_critic,
        )


# ======================================================================== #
# Original MLP-based actor-critic (unchanged behaviour)
# ======================================================================== #


class EncoderActorCritic(_EncoderActorCriticBase):
    """Standard MLP PPO policy augmented with an optional observation encoder."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        encoder_hidden_dims: Sequence[int] | None = None,
        share_encoder: bool | None = None,
        future_command_encoder: dict | None = None,
        **kwargs,
    ):
        # Filter out MoE-specific kwargs that may leak from shared config
        kwargs.pop("moe_config", None)
        if kwargs:
            print(
                "EncoderActorCritic.__init__ got unexpected arguments which will be ignored: "
                + ", ".join(kwargs.keys())
            )
        super().__init__()

        actor_encoded_dim = self._init_base(
            num_actor_obs, num_critic_obs, num_actions,
            actor_hidden_dims, critic_hidden_dims, activation,
            init_noise_std, noise_std_type, encoder_hidden_dims,
            share_encoder, future_command_encoder,
        )

        # Actor head (single MLP)
        self.actor = _build_mlp(actor_encoded_dim, actor_hidden_dims, self.num_actions, self._activation_ctor)

    # --------------------------------------------------------------------- #
    # Actor-Critic API (mirrors rsl_rl.modules.ActorCritic)
    # --------------------------------------------------------------------- #
    def update_distribution(self, observations: torch.Tensor):
        encoded = self._encode_actor_obs(observations)
        mean = self.actor(encoded)
        std = self._current_std().expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        if self.distribution is None:
            raise RuntimeError("Distribution is not initialized. Call update_distribution first.")
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        encoded = self._encode_actor_obs(observations)
        return self.actor(encoded)


# ======================================================================== #
# MoE (Mixture-of-Experts) actor-critic
# ======================================================================== #


class MoEActorCritic(_EncoderActorCriticBase):
    """PPO policy with a Mixture-of-Experts actor head.

    The encoder, FutureCommandProcessor, critic, and noise logic are
    inherited from ``_EncoderActorCriticBase`` and are identical to
    ``EncoderActorCritic``.  Only the actor head is replaced by *N*
    expert MLPs + a learned gating network.

    Routing uses **motion-conditioned soft top-k** with low-frequency
    candidate updates:

    * The gate network takes ``z_t`` (future-action latent from the
      command processor) as input — *not* the full encoded observation.
    * Every *M* rollout steps the top-k candidate experts are refreshed.
    * Between refreshes, soft weights ``softmax(logits[K] / T)`` are
      recomputed each step for smooth blending.

    Two routing modes are supported:

    * **supervised** (Phase A) – curriculum mask limits which experts the
      router may select; a CE auxiliary loss trains the gate.
    * **adaptive** (Phase B) – all experts available; a load-balancing
      loss prevents collapse.  New experts can be added dynamically.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        encoder_hidden_dims: Sequence[int] | None = None,
        share_encoder: bool | None = None,
        future_command_encoder: dict | None = None,
        moe_config: dict | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "MoEActorCritic.__init__ got unexpected arguments which will be ignored: "
                + ", ".join(kwargs.keys())
            )
        super().__init__()

        actor_encoded_dim = self._init_base(
            num_actor_obs, num_critic_obs, num_actions,
            actor_hidden_dims, critic_hidden_dims, activation,
            init_noise_std, noise_std_type, encoder_hidden_dims,
            share_encoder, future_command_encoder,
        )

        # ---- MoE configuration ----
        moe_config = moe_config or {}
        self._num_experts: int = int(moe_config.get("num_experts", 12))
        self._expert_hidden_dims: list[int] = list(moe_config.get("expert_hidden_dims", [512, 256]))
        gate_hidden_dims: Sequence[int] = list(moe_config.get("gate_hidden_dims", [256]))
        self._top_k: int = int(moe_config.get("top_k", 3))
        self._routing_loss_coef: float = float(moe_config.get("routing_loss_coef", 0.1))
        self._load_balance_loss_coef: float = float(moe_config.get("load_balance_loss_coef", 0.01))
        self._supervised_hard_ratio: float = float(moe_config.get("supervised_hard_ratio", 0.0))
        # 新 level 样本使用 hard routing 的比例（确保新 expert 学到对应特征）
        self._new_level_hard_ratio: float = float(moe_config.get("new_level_hard_ratio", 0.8))

        # ---- Soft top-k routing parameters ----
        self._candidate_update_interval: int = int(moe_config.get("candidate_update_interval", 10))
        self._temperature: float = float(moe_config.get("temperature", 0.5))
        self._logits_ema_alpha: float = float(moe_config.get("logits_ema_alpha", 0.8))
        self._use_zt_for_gate: bool = bool(moe_config.get("use_zt_for_gate", True))

        # ---- Phase B (dynamic expert addition) ----
        self._phase_b_enabled: bool = bool(moe_config.get("phase_b_enabled", False))
        self._new_expert_weight_cap: float = float(moe_config.get("new_expert_weight_cap", 0.3))
        self._new_expert_warmup_steps: int = int(moe_config.get("new_expert_warmup_steps", 5000))
        self._max_total_experts: int = int(moe_config.get("max_total_experts", 16))

        # CE coefficient for labeled samples in adaptive mode (keeps router grounded)
        self._adaptive_ce_coef: float = float(moe_config.get("adaptive_ce_coef", 0.05))

        # ---- Save dims for later expert addition ----
        self._actor_encoded_dim: int = actor_encoded_dim

        # ---- expert MLPs ----
        self.experts = nn.ModuleList([
            _build_mlp(actor_encoded_dim, self._expert_hidden_dims, num_actions, self._activation_ctor)
            for _ in range(self._num_experts)
        ])

        # ---- gating network (input = z_t if available, else encoded_obs) ----
        gate_input_dim = actor_encoded_dim
        if self._use_zt_for_gate and self._actor_command_processor is not None:
            zt_dim = self._actor_command_processor.latent_output_dim
            if zt_dim > 0:
                gate_input_dim = zt_dim
        self._gate_input_dim: int = gate_input_dim
        self.gate = _build_mlp(gate_input_dim, gate_hidden_dims, self._num_experts, self._activation_ctor)

        # ---- runtime state (set by runner / PPO) ----
        self._routing_mode: str = "supervised"      # "supervised" | "adaptive"
        self._level_labels: torch.Tensor | None = None  # (B,) int tensor, 1-based level
        self._max_active_experts: int = 1            # grows with curriculum level
        self._current_max_level: int = 1            # current highest unlocked level (for hard routing)
        self.moe_aux_loss: torch.Tensor = torch.tensor(0.0)

        # ---- soft top-k runtime buffers ----
        self._candidate_indices: torch.Tensor | None = None   # (B, k) cached candidates
        self._logits_ema: torch.Tensor | None = None           # (B, num_experts)
        self._rollout_step: int = 0

        # ---- Phase B runtime state ----
        self._new_expert_id: int = -1                # index of most recently added expert
        self._new_expert_step_counter: int = 0       # cold-start counter

        # ---- Routing diagnostics (exposed for MoEPPO difficulty tracking) ----
        self._last_routing_entropy: torch.Tensor | None = None   # (B,)
        self._last_routing_gap: torch.Tensor | None = None       # (B,) top1-top2 gap

        print(
            f"[MoEActorCritic] experts={self._num_experts}, "
            f"expert_dims={self._expert_hidden_dims}, gate_input_dim={gate_input_dim}, "
            f"top_k={self._top_k}, temperature={self._temperature}, "
            f"candidate_M={self._candidate_update_interval}, "
            f"use_zt_gate={self._use_zt_for_gate}, phase_b={self._phase_b_enabled}"
        )

    # ---- public setters called by runner ----

    def set_level_labels(self, labels: torch.Tensor) -> None:
        """Set per-sample difficulty level labels (1-based)."""
        self._level_labels = labels

    def set_routing_mode(self, mode: str) -> None:
        assert mode in ("supervised", "adaptive"), f"Unknown routing mode: {mode}"
        if mode != self._routing_mode:
            print(f"[MoEActorCritic] routing mode: {self._routing_mode} -> {mode}")
        self._routing_mode = mode

    def set_max_active_experts(self, n: int) -> None:
        n = max(1, min(n, self._num_experts))
        if n != self._max_active_experts:
            print(f"[MoEActorCritic] max_active_experts: {self._max_active_experts} -> {n}")
            # Invalidate cached candidates — k may have changed
            self._candidate_indices = None
            self._logits_ema = None
        self._max_active_experts = n

    def set_current_max_level(self, level: int) -> None:
        """Set current max curriculum level (for hard routing of new level samples)."""
        if level != self._current_max_level:
            print(f"[MoEActorCritic] current_max_level: {self._current_max_level} -> {level}")
        self._current_max_level = level

    def copy_expert_weights(self, from_idx: int, to_idx: int) -> None:
        """Copy weights from one expert to another (warm start for new level).
        
        When upgrading to a new level, copy the previous expert's weights to the
        new expert so it starts with basic motion capabilities instead of random init.
        """
        if 0 <= from_idx < self._num_experts and 0 <= to_idx < self._num_experts:
            self.experts[to_idx].load_state_dict(self.experts[from_idx].state_dict())
            print(f"[MoEActorCritic] Copied expert {from_idx} weights to expert {to_idx} (warm start)")
        else:
            print(f"[MoEActorCritic] Warning: invalid expert indices {from_idx} -> {to_idx}")

    def reset_rollout_state(self) -> None:
        """Reset low-frequency routing buffers (call at start of each rollout)."""
        self._candidate_indices = None
        self._logits_ema = None
        self._rollout_step = 0

    # ---- z_t extraction ----

    def _encode_actor_obs_with_zt(
        self, obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode actor observations and extract z_t (future latent) separately.

        Returns:
            encoded: Encoder output for expert networks (B, actor_encoded_dim).
            z_t: Future-command latent (B, latent_dim) or *None* when no
                 command processor / conv is configured.
        """
        processor = self._actor_command_processor
        z_t: torch.Tensor | None = None

        if processor is not None and hasattr(processor, "forward_with_latent"):
            cmd_dim = processor.input_dim
            command = obs[:, :cmd_dim]
            remainder = obs[:, cmd_dim:]
            processed, z_t_raw = processor.forward_with_latent(command)
            if z_t_raw.shape[-1] == 0:
                z_t = None
            else:
                z_t = z_t_raw
            obs_processed = (
                torch.cat([processed, remainder], dim=-1)
                if remainder.numel() > 0
                else processed
            )
        else:
            obs_processed = self._apply_command_processor(obs, processor)

        if self._share_encoder:
            encoded = self._shared_encoder(self._actor_adapter(obs_processed))
        else:
            encoded = self._actor_encoder(obs_processed)

        return encoded, z_t

    # ---- MoE forward (soft top-k routing) ----

    def _moe_forward(
        self, encoded: torch.Tensor, z_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run MoE actor forward pass with soft top-k routing.

        Args:
            encoded: Encoder output (B, actor_encoded_dim).
            z_t: Optional future-command latent for gating (B, latent_dim).

        Returns:
            action_mean: Weighted expert output (B, num_actions).
        """
        import torch.nn.functional as F  # noqa: N812

        B = encoded.shape[0]
        device = encoded.device

        # 1) Gate logits — use z_t when available and configured
        if z_t is not None and self._use_zt_for_gate:
            gate_input = z_t
        else:
            gate_input = encoded
        gate_logits = self.gate(gate_input)  # (B, num_experts)

        # 2) Curriculum mask: only allow unlocked experts
        if self._max_active_experts < self._num_experts:
            gate_logits = gate_logits.clone()
            gate_logits[:, self._max_active_experts :] = float("-inf")

        # 3) Determine k and candidates
        k = max(1, min(self._top_k, self._max_active_experts))
        is_rollout = not torch.is_grad_enabled()

        if is_rollout:
            # EMA smoothing on logits (rollout only)
            if self._logits_ema_alpha < 1.0:
                if (
                    self._logits_ema is not None
                    and self._logits_ema.shape == gate_logits.shape
                ):
                    gate_logits = (
                        self._logits_ema_alpha * gate_logits
                        + (1.0 - self._logits_ema_alpha) * self._logits_ema
                    )
                self._logits_ema = gate_logits.detach().clone()

            # Low-frequency candidate update
            need_update = (
                self._rollout_step % self._candidate_update_interval == 0
                or self._candidate_indices is None
                or self._candidate_indices.shape[0] != B
                or self._candidate_indices.shape[-1] != k  # k changed (level up)
            )
            if need_update:
                _, candidates = torch.topk(gate_logits, k, dim=-1)
                self._candidate_indices = candidates
            candidates = self._candidate_indices
            self._rollout_step += 1
        else:
            # Training: fresh candidate selection each mini-batch
            _, candidates = torch.topk(gate_logits, k, dim=-1)

        # 4) Soft weights with temperature
        candidate_logits = gate_logits.gather(1, candidates)  # (B, k)
        weights = F.softmax(candidate_logits / self._temperature, dim=-1)  # (B, k)

        # 5) Phase B cold-start: cap new expert weight
        if (
            self._new_expert_id >= 0
            and self._new_expert_step_counter < self._new_expert_warmup_steps
        ):
            is_new = candidates == self._new_expert_id
            if is_new.any():
                weights = weights.clone()
                weights[is_new] = weights[is_new].clamp(max=self._new_expert_weight_cap)
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
            if is_rollout:
                self._new_expert_step_counter += 1

        # 6) Compute only candidate expert outputs (sparse, saves VRAM)
        unique_ids = torch.unique(candidates)
        unique_list = unique_ids.tolist()
        expert_outs_list = [self.experts[i](encoded) for i in unique_list]
        stacked = torch.stack(expert_outs_list, dim=0)  # (U, B, A)
        stacked = stacked.permute(1, 0, 2)              # (B, U, A)

        # Map candidate indices to positions in the stacked tensor
        assert candidates is not None  # guaranteed by topk above
        mapped = torch.searchsorted(unique_ids, candidates.reshape(-1))
        mapped = mapped.reshape(B, k)  # (B, k)

        gathered = torch.gather(
            stacked, 1, mapped.unsqueeze(-1).expand(-1, -1, self.num_actions),
        )  # (B, k, A)

        # 7) Weighted sum of expert outputs
        action_mean = (weights.unsqueeze(-1) * gathered).sum(dim=1)  # (B, A)

        # 7.5) Hard routing for new level samples (training only)
        # 新 level 样本的一定比例强制只用对应 expert，确保新 expert 学到对应特征
        if (
            torch.is_grad_enabled()
            and self._new_level_hard_ratio > 0
            and self._level_labels is not None
            and self._current_max_level > 1
        ):
            # 识别新 level 样本：level == current_max_level
            is_new_level = self._level_labels == self._current_max_level
            if is_new_level.any():
                # 随机选择哪些新 level 样本使用 hard routing
                hard_mask = is_new_level & (
                    torch.rand(B, device=device) < self._new_level_hard_ratio
                )
                if hard_mask.any():
                    # Hard routing: 强制只用对应 expert
                    target_expert_idx = min(
                        self._current_max_level - 1, self._num_experts - 1
                    )
                    hard_out = self.experts[target_expert_idx](encoded[hard_mask])
                    action_mean = action_mean.clone()
                    action_mean[hard_mask] = hard_out

        # 8) Routing diagnostics (for Phase B difficulty tracking)
        with torch.no_grad():
            gate_probs = F.softmax(gate_logits, dim=-1)
            log_probs = torch.log(gate_probs + 1e-8)
            self._last_routing_entropy = -(gate_probs * log_probs).sum(dim=-1)
            top2_vals, _ = torch.topk(
                gate_probs, min(2, gate_probs.shape[-1]), dim=-1,
            )
            if top2_vals.shape[-1] >= 2:
                self._last_routing_gap = top2_vals[:, 0] - top2_vals[:, 1]
            else:
                self._last_routing_gap = torch.ones(B, device=device)

        # 9) Auxiliary loss (only during training, not rollout)
        if torch.is_grad_enabled():
            self._compute_aux_loss(gate_logits)
        else:
            self.moe_aux_loss = torch.tensor(0.0, device=device)

        return action_mean

    # ---- auxiliary loss ----

    def _compute_aux_loss(self, gate_logits: torch.Tensor) -> None:
        """Compute MoE auxiliary loss (CE for supervised, load-balance for adaptive).
        
        NOTE: expert_idx must be clamped to _max_active_experts - 1 (not _num_experts - 1)
        because gate_logits[:, _max_active_experts:] are set to -inf by the curriculum mask.
        Using a larger clamp would cause cross_entropy to compute -log(0) = inf.
        """
        import torch.nn.functional as F  # noqa: N812

        # Clamp to max_active_experts - 1 to avoid indexing into -inf logits
        max_valid_expert = max(0, self._max_active_experts - 1)

        if self._routing_mode == "supervised" and self._level_labels is not None:
            expert_idx = (self._level_labels - 1).clamp(0, max_valid_expert).long()
            routing_loss = F.cross_entropy(gate_logits, expert_idx)
            self.moe_aux_loss = self._routing_loss_coef * routing_loss
        else:
            gate_probs = F.softmax(gate_logits, dim=-1)
            load_balance = self._load_balance_loss(gate_probs)
            self.moe_aux_loss = self._load_balance_loss_coef * load_balance

            # CE supervision on labeled samples in adaptive mode (keeps router grounded)
            if self._level_labels is not None:
                labeled = self._level_labels > 0
                if labeled.any():
                    idx = (self._level_labels[labeled] - 1).clamp(0, max_valid_expert).long()
                    ce = F.cross_entropy(gate_logits[labeled], idx)
                    self.moe_aux_loss = self.moe_aux_loss + self._adaptive_ce_coef * ce

    # ---- load-balance loss (Switch-Transformer style, vectorized) ----

    @staticmethod
    def _load_balance_loss(gate_probs: torch.Tensor) -> torch.Tensor:
        n = gate_probs.shape[-1]
        top1 = gate_probs.argmax(dim=-1)
        counts = torch.bincount(top1, minlength=n).float()
        f = counts / max(top1.shape[0], 1)
        P = gate_probs.mean(dim=0)
        return n * (f * P).sum()

    # ---- Phase B: dynamic expert addition ----

    def add_expert(self) -> int:
        """Dynamically add a new expert MLP and expand the gate output.

        Returns:
            Index of the newly added expert, or -1 if at max capacity.
        """
        if self._num_experts >= self._max_total_experts:
            print(
                f"[MoEActorCritic] Cannot add expert: already at max ({self._max_total_experts})"
            )
            return -1

        new_id = self._num_experts
        device = next(self.parameters()).device

        # 1) New expert MLP
        new_expert = _build_mlp(
            self._actor_encoded_dim, self._expert_hidden_dims,
            self.num_actions, self._activation_ctor,
        ).to(device)
        self.experts.append(new_expert)

        # 2) Expand gate output: replace last Linear layer
        last_linear = self.gate[-1]
        assert isinstance(last_linear, nn.Linear), (
            f"Expected last gate layer to be Linear, got {type(last_linear)}"
        )
        old_out = last_linear.out_features
        new_out = old_out + 1
        has_bias = last_linear.bias is not None
        new_linear = nn.Linear(last_linear.in_features, new_out, bias=has_bias).to(device)

        with torch.no_grad():
            new_linear.weight[:old_out] = last_linear.weight
            nn.init.xavier_uniform_(new_linear.weight[old_out : old_out + 1])
            if has_bias and new_linear.bias is not None:
                new_linear.bias[:old_out] = last_linear.bias  # type: ignore[index]
                new_linear.bias[old_out] = 0.0

        self.gate[-1] = new_linear

        # 3) Update bookkeeping
        self._num_experts = new_out
        self._new_expert_id = new_id
        self._new_expert_step_counter = 0

        # Invalidate cached candidates
        self._candidate_indices = None
        self._logits_ema = None

        print(
            f"[MoEActorCritic] Added expert {new_id} "
            f"(total={self._num_experts}, warmup={self._new_expert_warmup_steps} steps)"
        )
        return new_id

    # ---- Actor-Critic API ----

    def update_distribution(self, observations: torch.Tensor):
        encoded, z_t = self._encode_actor_obs_with_zt(observations)
        mean = self._moe_forward(encoded, z_t)
        std = self._current_std().expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        if self.distribution is None:
            raise RuntimeError("Distribution is not initialized.")
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Deterministic inference – uses soft top-k gating with z_t."""
        encoded, z_t = self._encode_actor_obs_with_zt(observations)
        return self._moe_forward(encoded, z_t)

    def act_inference_onnx(self, observations: torch.Tensor) -> torch.Tensor:
        """ONNX-compatible deterministic inference.
        
        Unlike _moe_forward, this avoids Python loops and searchsorted by
        computing all active experts upfront (slightly less memory efficient,
        but fully traceable for ONNX export).
        """
        import torch.nn.functional as F  # noqa: N812

        encoded, z_t = self._encode_actor_obs_with_zt(observations)
        B = encoded.shape[0]

        # 1) Gate logits
        if z_t is not None and self._use_zt_for_gate:
            gate_input = z_t
        else:
            gate_input = encoded
        gate_logits = self.gate(gate_input)  # (B, num_experts)

        # 2) Curriculum mask: only allow unlocked experts
        if self._max_active_experts < self._num_experts:
            gate_logits = gate_logits.clone()
            gate_logits[:, self._max_active_experts:] = float("-inf")

        # 3) Top-k selection
        k = max(1, min(self._top_k, self._max_active_experts))
        _, candidates = torch.topk(gate_logits, k, dim=-1)  # (B, k)

        # 4) Soft weights
        candidate_logits = gate_logits.gather(1, candidates)
        weights = F.softmax(candidate_logits / self._temperature, dim=-1)  # (B, k)

        # 5) Compute ALL active expert outputs (ONNX-friendly: no Python loop over dynamic indices)
        # Stack all expert outputs: (num_experts, B, A)
        all_expert_outs = torch.stack([expert(encoded) for expert in self.experts], dim=0)
        all_expert_outs = all_expert_outs.permute(1, 0, 2)  # (B, num_experts, A)

        # 6) Gather candidate expert outputs using indices
        # candidates: (B, k) -> expand to (B, k, A)
        gathered = torch.gather(
            all_expert_outs, 1,
            candidates.unsqueeze(-1).expand(-1, -1, self.num_actions)
        )  # (B, k, A)

        # 7) Weighted sum
        action_mean = (weights.unsqueeze(-1) * gathered).sum(dim=1)  # (B, A)
        return action_mean




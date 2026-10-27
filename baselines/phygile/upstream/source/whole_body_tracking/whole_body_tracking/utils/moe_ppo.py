"""MoE-aware PPO algorithm with level-label side buffer and Phase B difficulty tracking.

Drop-in replacement for ``rsl_rl.algorithms.ppo.PPO``.  The differences are:

1.  A parallel ``level_labels`` buffer is maintained alongside the
    standard ``RolloutStorage``.
2.  During ``update()`` the level labels are fed to the policy so that
    the MoE gating network can compute its supervised / load-balance
    auxiliary loss.
3.  The auxiliary loss (``policy.moe_aux_loss``) is added to the total
    PPO loss before the backward pass.
4.  **Phase B**: Per-file difficulty tracking (error EMA, routing entropy,
    top1-top2 gap) with ``check_difficulty()`` to trigger new expert
    addition or LR boost when at max capacity.
    ``rebuild_optimizer_for_new_expert()`` creates separate parameter
    groups with different learning rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage import RolloutStorage


@dataclass
class DifficultyThresholds:
    """Thresholds for Phase B new-expert trigger (per-file, 2-of-3 conditions)."""

    err: float = 0.15           # per-file tracking error EMA threshold
    entropy: float = 0.8        # per-file routing entropy EMA threshold
    gap: float = 0.15           # per-file top1-top2 gap threshold (below = no dominant expert)
    sustained_windows: int = 30  # consecutive check windows a file must stay difficult
    trigger_ratio: float = 0.1  # fraction of *files* that must be difficult to trigger
    check_interval: int = 500   # only check every N iterations


class MoEPPO(PPO):
    """PPO extended with MoE auxiliary-loss support and Phase B difficulty tracking.

    *All* standard PPO parameters are accepted (and forwarded to the
    base class); any additional / unknown keyword arguments are silently
    ignored so that the same ``algorithm`` config dict can be re-used
    between ``PPO`` and ``MoEPPO`` without modification.
    """

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        # Accept (and ignore) any extra keys that may leak from config
        **_extra,
    ):
        if _extra:
            print(f"[MoEPPO] ignoring extra kwargs: {list(_extra.keys())}")
        super().__init__(
            policy,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
        )

        # ---- MoE level-label side buffer ----
        self._level_buffer: torch.Tensor | None = None
        self._current_level_labels: torch.Tensor | None = None

        # Exposed for logging
        self.mean_moe_aux_loss: float = 0.0

        # ---- Phase B stability: error clamp, critic warmup, surrogate ramp ----
        self._max_value_err: float = 0.0  # >0: clamp per-sample value error before squaring
        self._critic_warmup_iters: int = 0
        self._surrogate_ramp_iters: int = 0
        self._phase_b_start_iter: int = 0
        self._phase_b_current_iter: int = 0

        # ---- Phase B per-file difficulty tracking ----
        self._file_indices_getter = None  # callable → torch.Tensor (num_envs,)
        self._file_error_ema: torch.Tensor | None = None        # (num_files,)
        self._file_entropy_ema: torch.Tensor | None = None       # (num_files,)
        self._file_gap_ema: torch.Tensor | None = None           # (num_files,)
        self._file_difficulty_counter: torch.Tensor | None = None  # (num_files,)
        self._num_tracked_files: int = 0
        self._difficulty_ema_alpha: float = 0.995

        # ---- Phase B LR boost state (when at max expert capacity) ----
        self._boost_active: bool = False
        self._boost_until_iter: int = 0
        self._pre_boost_lrs: list[float] = []

        # Exposed diagnostics for logging
        self.mean_difficulty_error: float = 0.0
        self.mean_difficulty_entropy: float = 0.0
        self.mean_difficulty_gap: float = 0.0
        self.difficult_file_ratio: float = 0.0

    # ------------------------------------------------------------------ #
    # Storage initialisation
    # ------------------------------------------------------------------ #

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        super().init_storage(
            num_envs, num_transitions_per_env,
            actor_obs_shape, critic_obs_shape, action_shape,
        )
        # Parallel level buffer – same first two dims as observations
        self._level_buffer = torch.zeros(
            num_transitions_per_env, num_envs,
            dtype=torch.long, device=self.device,
        )
        # Per-file tracking is initialised lazily via init_file_tracking()

    # ------------------------------------------------------------------ #
    # Phase B per-file tracking setup
    # ------------------------------------------------------------------ #

    def set_file_indices_getter(self, getter) -> None:
        """Set a callable that returns per-env file indices (torch.Tensor or None)."""
        self._file_indices_getter = getter

    def configure_phase_b_stability(
        self,
        max_value_err: float = 10.0,
        critic_warmup_iters: int = 3000,
        surrogate_ramp_iters: int = 5000,
        start_iter: int = 0,
    ) -> None:
        """Configure Phase B stability measures.

        Args:
            max_value_err: Clamp per-sample value prediction error before squaring.
                Keeps MSE behavior but bounds gradient at 2*max_value_err.
            critic_warmup_iters: First N iters only train critic (surrogate=0).
            surrogate_ramp_iters: After warmup, linearly ramp surrogate weight 0→1.
            start_iter: The iteration at which Phase B starts.
        """
        self._max_value_err = max_value_err
        self._critic_warmup_iters = critic_warmup_iters
        self._surrogate_ramp_iters = surrogate_ramp_iters
        self._phase_b_start_iter = start_iter
        self._phase_b_current_iter = start_iter
        print(
            f"[MoEPPO] Phase B stability: max_value_err={max_value_err}, "
            f"critic_warmup={critic_warmup_iters}, surrogate_ramp={surrogate_ramp_iters} "
            f"iters from iter {start_iter}"
        )

    def init_file_tracking(self, num_files: int) -> None:
        """Initialise per-file difficulty tracking buffers (Phase B only)."""
        self._num_tracked_files = num_files
        self._file_error_ema = torch.zeros(num_files, device=self.device)
        self._file_entropy_ema = torch.zeros(num_files, device=self.device)
        self._file_gap_ema = torch.ones(num_files, device=self.device)
        self._file_difficulty_counter = torch.zeros(
            num_files, dtype=torch.long, device=self.device,
        )
        print(f"[MoEPPO] Initialised per-file difficulty tracking for {num_files} files")

    # ------------------------------------------------------------------ #
    # Level-label auto-fetch (allows parent learn() to work unmodified)
    # ------------------------------------------------------------------ #

    def set_level_label_getter(self, getter) -> None:
        """Set a callable that returns per-env level labels (torch.Tensor or None)."""
        self._level_label_getter = getter

    # ------------------------------------------------------------------ #
    # Rollout collection (called under torch.inference_mode)
    # ------------------------------------------------------------------ #

    def act(self, obs, critic_obs, level_labels=None):
        """Extends ``PPO.act`` to accept optional *level_labels*.

        If *level_labels* is not provided, attempts to auto-fetch via the
        getter registered by ``set_level_label_getter``.
        """
        if level_labels is None and hasattr(self, "_level_label_getter"):
            level_labels = self._level_label_getter()
        if level_labels is not None:
            self._current_level_labels = level_labels
        return super().act(obs, critic_obs)

    def process_env_step(self, rewards, dones, infos):
        # Write level labels at the *current* step (before super increments it)
        if (
            self._current_level_labels is not None
            and self._level_buffer is not None
            and self.storage is not None
            and self.storage.step < self._level_buffer.shape[0]
        ):
            self._level_buffer[self.storage.step] = self._current_level_labels

        # ---- Phase B: update difficulty tracking EMA ----
        self._update_difficulty_tracking(rewards, dones, infos)

        super().process_env_step(rewards, dones, infos)

    def _update_difficulty_tracking(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
    ) -> None:
        """Aggregate per-env metrics into per-file EMA via scatter."""
        if self._file_error_ema is None or self._file_indices_getter is None:
            return

        file_indices = self._file_indices_getter()
        if file_indices is None:
            return

        num_envs = min(rewards.shape[0], file_indices.shape[0])
        device = self._file_error_ema.device
        num_files = self._num_tracked_files

        # Skip done envs — their file_indices may already point to a new file
        alive_mask = dones[:num_envs] <= 0
        if not alive_mask.any():
            return
        alive_idx = alive_mask.nonzero(as_tuple=False).squeeze(-1)
        alive_files = file_indices[alive_idx].to(device)

        # Clamp to valid range
        alive_files = alive_files.clamp(0, num_files - 1)

        # --- Per-env metrics ---
        ac = self.policy
        if "episode" in infos and "tracking_error" in infos["episode"]:
            err_raw = infos["episode"]["tracking_error"]
            if not isinstance(err_raw, torch.Tensor):
                err_raw = torch.tensor(err_raw, device=device)
            err = err_raw[:num_envs][alive_idx].to(device)
        else:
            err = (-rewards[:num_envs][alive_idx]).clamp(min=0.0).to(device)

        ent = None
        if hasattr(ac, "_last_routing_entropy") and ac._last_routing_entropy is not None:
            ent = ac._last_routing_entropy[:num_envs][alive_idx].to(device)
        gap = None
        if hasattr(ac, "_last_routing_gap") and ac._last_routing_gap is not None:
            gap = ac._last_routing_gap[:num_envs][alive_idx].to(device)

        # --- Scatter-aggregate to per-file means ---
        ones = torch.ones_like(alive_files, dtype=torch.float)
        file_count = torch.zeros(num_files, device=device)
        file_count.scatter_add_(0, alive_files, ones)
        active = file_count > 0
        safe_count = file_count.clamp(min=1)

        alpha = self._difficulty_ema_alpha

        file_err_sum = torch.zeros(num_files, device=device)
        file_err_sum.scatter_add_(0, alive_files, err)
        mean_err = file_err_sum / safe_count
        self._file_error_ema[active] = (
            alpha * self._file_error_ema[active] + (1.0 - alpha) * mean_err[active]
        )

        if ent is not None and self._file_entropy_ema is not None:
            file_ent_sum = torch.zeros(num_files, device=device)
            file_ent_sum.scatter_add_(0, alive_files, ent)
            mean_ent = file_ent_sum / safe_count
            self._file_entropy_ema[active] = (
                alpha * self._file_entropy_ema[active] + (1.0 - alpha) * mean_ent[active]
            )

        if gap is not None and self._file_gap_ema is not None:
            file_gap_sum = torch.zeros(num_files, device=device)
            file_gap_sum.scatter_add_(0, alive_files, gap)
            mean_gap = file_gap_sum / safe_count
            self._file_gap_ema[active] = (
                alpha * self._file_gap_ema[active] + (1.0 - alpha) * mean_gap[active]
            )

    # ------------------------------------------------------------------ #
    # Phase B: difficulty check
    # ------------------------------------------------------------------ #

    def check_difficulty(self, thresholds: DifficultyThresholds) -> str:
        """Check per-file difficulty with 2-of-3 conditions.

        Conditions (any 2 sufficient):
          1. file error EMA  > thresholds.err
          2. file entropy EMA > thresholds.entropy
          3. file gap EMA     < thresholds.gap   (no dominant expert)

        Returns:
          ``"add"``   – difficulty triggered and experts can be added
          ``"boost"`` – difficulty triggered but at max expert capacity
          ``"none"``  – not triggered
        """
        if (
            self._file_error_ema is None
            or self._file_entropy_ema is None
            or self._file_gap_ema is None
            or self._file_difficulty_counter is None
        ):
            return "none"

        cond_err = self._file_error_ema > thresholds.err
        cond_ent = self._file_entropy_ema > thresholds.entropy
        cond_gap = self._file_gap_ema < thresholds.gap

        difficult = (cond_err.float() + cond_ent.float() + cond_gap.float()) >= 2.0

        self._file_difficulty_counter[difficult] += 1
        self._file_difficulty_counter[~difficult] = 0

        sustained = self._file_difficulty_counter > thresholds.sustained_windows
        ratio = sustained.float().mean() if sustained.numel() > 0 else 0.0

        # Expose diagnostics for logging
        self.mean_difficulty_error = float(self._file_error_ema.mean().item())
        self.mean_difficulty_entropy = float(self._file_entropy_ema.mean().item())
        self.mean_difficulty_gap = float(self._file_gap_ema.mean().item())
        self.difficult_file_ratio = float(ratio)

        if float(ratio) <= thresholds.trigger_ratio:
            return "none"

        ac = self.policy
        at_max = hasattr(ac, "_num_experts") and hasattr(ac, "_max_total_experts") and (
            ac._num_experts >= ac._max_total_experts
        )
        return "boost" if at_max else "add"

    # ------------------------------------------------------------------ #
    # Phase B: optimizer rebuild with per-group learning rates
    # ------------------------------------------------------------------ #

    def rebuild_optimizer_for_new_expert(
        self,
        new_expert_lr_scale: float = 3.0,
        old_expert_lr_scale: float = 0.1,
    ) -> None:
        """Rebuild optimizer with separate param groups after expert addition.

        * encoder + critic + gate: ``learning_rate`` (unchanged)
        * old experts: ``learning_rate * old_expert_lr_scale``
        * new expert (last in ``experts``): ``learning_rate * new_expert_lr_scale``
        """
        ac = self.policy
        new_expert_idx = len(ac.experts) - 1  # type: ignore[arg-type]

        # Collect parameter sets
        expert_params_old: list[torch.nn.Parameter] = []
        expert_params_new: list[torch.nn.Parameter] = []
        other_params: list[torch.nn.Parameter] = []

        expert_param_ids: set[int] = set()
        for i, expert in enumerate(ac.experts):  # type: ignore[union-attr]
            for p in expert.parameters():
                expert_param_ids.add(id(p))
                if i == new_expert_idx:
                    expert_params_new.append(p)
                else:
                    expert_params_old.append(p)

        for p in ac.parameters():
            if id(p) not in expert_param_ids:
                other_params.append(p)

        param_groups = [
            {"params": other_params, "lr": self.learning_rate},
            {"params": expert_params_old, "lr": self.learning_rate * old_expert_lr_scale},
            {"params": expert_params_new, "lr": self.learning_rate * new_expert_lr_scale},
        ]
        self.optimizer = optim.Adam(param_groups)
        print(
            f"[MoEPPO] Rebuilt optimizer: base_lr={self.learning_rate:.5f}, "
            f"old_expert_lr={self.learning_rate * old_expert_lr_scale:.5f}, "
            f"new_expert_lr={self.learning_rate * new_expert_lr_scale:.5f}"
        )

    # ------------------------------------------------------------------ #
    # Phase B: temporary LR boost (when at max expert capacity)
    # ------------------------------------------------------------------ #

    def boost_expert_lr(
        self,
        current_iter: int,
        scale: float = 2.0,
        duration_iters: int = 3000,
    ) -> None:
        """Temporarily multiply all param-group LRs by *scale*.

        Automatically restores after *duration_iters* via
        ``maybe_restore_boost``.
        """
        if self._boost_active:
            return
        self._pre_boost_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        for pg in self.optimizer.param_groups:
            pg["lr"] *= scale
        self._boost_active = True
        self._boost_until_iter = current_iter + duration_iters
        print(
            f"[MoEPPO] LR boosted x{scale} for {duration_iters} iters "
            f"(restore at iter {self._boost_until_iter})"
        )

    def maybe_restore_boost(self, current_iter: int) -> None:
        """Restore original LRs if the boost window has elapsed."""
        if not self._boost_active:
            return
        if current_iter < self._boost_until_iter:
            return
        for pg, lr in zip(self.optimizer.param_groups, self._pre_boost_lrs):
            pg["lr"] = lr
        self._boost_active = False
        self._pre_boost_lrs = []
        print(f"[MoEPPO] LR boost restored at iter {current_iter}")

    # ------------------------------------------------------------------ #
    # PPO update – identical to base except for MoE auxiliary loss
    # ------------------------------------------------------------------ #

    def update(self):  # noqa: C901 – intentional mirror of upstream PPO.update
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_moe_aux = 0.0

        # Flatten level buffer the same way the storage flattens observations
        level_flat: torch.Tensor | None = None
        if self._level_buffer is not None:
            level_flat = self._level_buffer.flatten(0, 1)  # (T*N,)

        # ---- replicate the standard mini-batch generator inline so we
        #      can access the *same* random indices for level_flat ----
        storage = self.storage
        batch_size = storage.num_envs * storage.num_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches
        indices = torch.randperm(
            self.num_mini_batches * mini_batch_size,
            requires_grad=False, device=self.device,
        )

        observations = storage.observations.flatten(0, 1)
        if storage.privileged_observations is not None:
            critic_observations = storage.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
        actions = storage.actions.flatten(0, 1)
        values = storage.values.flatten(0, 1)
        returns = storage.returns.flatten(0, 1)
        old_actions_log_prob = storage.actions_log_prob.flatten(0, 1)
        advantages = storage.advantages.flatten(0, 1)
        old_mu = storage.mu.flatten(0, 1)
        old_sigma = storage.sigma.flatten(0, 1)

        for _epoch in range(self.num_learning_epochs):
            for i in range(self.num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_obs_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # ---- MoE: inject level labels ----
                if level_flat is not None and hasattr(self.policy, "set_level_labels"):
                    self.policy.set_level_labels(level_flat[batch_idx])

                # ---- standard PPO forward ----
                self.policy.act(obs_batch)
                actions_log_prob_batch_new = self.policy.get_actions_log_prob(actions_batch)
                value_batch = self.policy.evaluate(critic_obs_batch)
                mu_batch = self.policy.action_mean
                sigma_batch = self.policy.action_std
                entropy_batch = self.policy.entropy

                # KL-based LR adaptation
                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1e-5)
                            + (old_sigma_batch.square() + (old_mu_batch - mu_batch).square())
                            / (2.0 * sigma_batch.square())
                            - 0.5,
                            axis=-1,
                        )
                        kl_mean = kl.mean()
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        for pg in self.optimizer.param_groups:
                            pg["lr"] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(
                    actions_log_prob_batch_new - torch.squeeze(old_actions_log_prob_batch)
                )
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value loss (MSE, with optional error clamping for Phase B stability)
                # Clamp the prediction *error* instead of the loss itself,
                # so gradients are preserved (bounded at 2*max_err) rather than
                # zeroed out by a hard loss clamp.
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (
                        value_batch - target_values_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    diff_raw = value_batch - returns_batch
                    diff_clip = value_clipped - returns_batch
                    if self._max_value_err > 0:
                        diff_raw = diff_raw.clamp(-self._max_value_err, self._max_value_err)
                        diff_clip = diff_clip.clamp(-self._max_value_err, self._max_value_err)
                    value_losses = diff_raw.pow(2)
                    value_losses_clipped = diff_clip.pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    diff = value_batch - returns_batch
                    if self._max_value_err > 0:
                        diff = diff.clamp(-self._max_value_err, self._max_value_err)
                    value_loss = diff.pow(2).mean()

                # ---- MoE auxiliary loss ----
                moe_aux = getattr(self.policy, "moe_aux_loss", 0.0)
                if isinstance(moe_aux, (int, float)):
                    moe_aux = torch.tensor(moe_aux, device=self.device)

                # Phase B surrogate ramp: warmup (weight=0) → linear ramp → full (weight=1)
                elapsed_b = self._phase_b_current_iter - self._phase_b_start_iter
                if self._critic_warmup_iters > 0 and elapsed_b < self._critic_warmup_iters:
                    surr_weight = 0.0
                elif self._surrogate_ramp_iters > 0:
                    ramp_elapsed = elapsed_b - self._critic_warmup_iters
                    surr_weight = min(1.0, max(0.0, ramp_elapsed / self._surrogate_ramp_iters))
                else:
                    surr_weight = 1.0
                effective_surrogate = surrogate_loss * surr_weight

                loss = (
                    effective_surrogate
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean()
                    + moe_aux
                )

                # Gradient step (skip if loss is non-finite to avoid NaN poisoning)
                if not torch.isfinite(loss):
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy_batch.mean().item()
                mean_moe_aux += moe_aux.item() if hasattr(moe_aux, "item") else float(moe_aux)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_moe_aux /= num_updates

        # Exposed for logging by the runner
        self.mean_moe_aux_loss = mean_moe_aux

        # Advance Phase B iteration counter
        self._phase_b_current_iter += 1

        # Reset buffers
        storage.clear()
        if self._level_buffer is not None:
            self._level_buffer.zero_()

        # Return dict matching rsl_rl PPO.update() contract
        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "moe_aux": mean_moe_aux,
        }

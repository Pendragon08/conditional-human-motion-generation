# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os

import onnx
import torch

from isaaclab.envs import ManagerBasedRLEnv

from whole_body_tracking.tasks.tracking.mdp import MotionCommand


def _resize_normalizer(normalizer, target_dim: int) -> None:
    """Pad or truncate normalizer stats to match target_dim."""
    if normalizer is None or not hasattr(normalizer, "_mean") or not hasattr(normalizer, "_std"):
        return
    try:
        current_dim = int(normalizer._mean.shape[-1])
    except Exception:
        return
    if current_dim == target_dim:
        return
    device = normalizer._mean.device
    if current_dim < target_dim:
        pad = target_dim - current_dim
        normalizer._mean = torch.cat(
            [normalizer._mean, torch.zeros(pad, device=device, dtype=normalizer._mean.dtype)], dim=-1
        )
        normalizer._std = torch.cat(
            [normalizer._std, torch.ones(pad, device=device, dtype=normalizer._std.dtype)], dim=-1
        )
    else:
        normalizer._mean = normalizer._mean[..., :target_dim]
        normalizer._std = normalizer._std[..., :target_dim]


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(torch.nn.Module):
    """Exporter for motion policy with full encoder support (including conv encoder and MoE)."""

    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        cmd: MotionCommand = env.command_manager.get_term("motion")

        # Motion data for replay
        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

        # Check if policy is MoE (has experts instead of actor)
        self._is_moe = hasattr(actor_critic, "experts") and hasattr(actor_critic, "gate")
        
        # Check if policy has EncoderActorCritic structure with conv encoder
        self._has_encoder = hasattr(actor_critic, "_actor_command_processor")
        
        if self._is_moe:
            # MoE policy: copy the entire actor_critic for inference
            self._moe_policy = copy.deepcopy(actor_critic)
            self._actor_obs_dim = actor_critic._actor_obs_dim
        elif self._has_encoder:
            # Copy full policy components for EncoderActorCritic
            self.actor = copy.deepcopy(actor_critic.actor)
            self._actor_command_processor = copy.deepcopy(actor_critic._actor_command_processor)
            
            # Copy encoder if exists
            if hasattr(actor_critic, "_share_encoder") and actor_critic._share_encoder:
                self._shared_encoder = copy.deepcopy(actor_critic._shared_encoder)
                self._actor_adapter = copy.deepcopy(actor_critic._actor_adapter)
                self._share_encoder = True
            else:
                if hasattr(actor_critic, "_actor_encoder"):
                    self._actor_encoder = copy.deepcopy(actor_critic._actor_encoder)
                self._share_encoder = False
            
            # Get raw observation dimension
            self._actor_obs_dim = actor_critic._actor_obs_dim
        else:
            # Standard policy without encoder
            if hasattr(actor_critic, "actor"):
                self.actor = copy.deepcopy(actor_critic.actor)
            else:
                raise ValueError("Policy does not have an actor module.")
            self._actor_obs_dim = int(self.actor[0].in_features)

        # Copy normalizer
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def _encode_actor_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply command processor and encoder to observations."""
        if self._actor_command_processor is not None:
            cmd_dim = self._actor_command_processor.input_dim
            command = obs[:, :cmd_dim]
            remainder = obs[:, cmd_dim:]
            processed = self._actor_command_processor(command)
            if remainder.numel() > 0:
                obs = torch.cat([processed, remainder], dim=-1)
            else:
                obs = processed
        
        if self._share_encoder:
            scaled = self._actor_adapter(obs)
            return self._shared_encoder(scaled)
        elif hasattr(self, "_actor_encoder"):
            return self._actor_encoder(obs)
        return obs

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        
        # Normalize observations
        x_normalized = self.normalizer(x)
        
        if self._is_moe:
            # MoE policy: use ONNX-compatible inference (avoids searchsorted etc.)
            actions = self._moe_policy.act_inference_onnx(x_normalized)
        elif self._has_encoder:
            # EncoderActorCritic: encode then pass through actor
            x_encoded = self._encode_actor_obs(x_normalized)
            actions = self.actor(x_encoded)
        else:
            # Standard policy
            actions = self.actor(x_normalized)
        
        return (
            actions,
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.to("cpu")
        
        # Use raw observation dimension (before encoding)
        obs_dim = self._actor_obs_dim
        _resize_normalizer(self.normalizer, obs_dim)
        
        if hasattr(self.normalizer, "to"):
            try:
                self.normalizer.to("cpu")
            except Exception:
                pass
        
        obs = torch.zeros(1, obs_dim)
        time_step = torch.zeros(1, 1)
        
        print(f"[INFO] Exporting ONNX with input obs dim: {obs_dim}")
        
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)

    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)
    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)

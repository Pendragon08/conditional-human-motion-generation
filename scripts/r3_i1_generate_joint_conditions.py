"""Inference-only multi-condition OmniControl experiment; no post-retarget IK.

The same pretrained diffusion model, seed, text and original kick target are
used in both arms. This arm adds *pre-generation* pelvis, stance-foot, guard
and recovery keyframes. They are designed conditions, not measured contacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import sample.generate as official
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from diffusion.gaussian_diffusion import GaussianDiffusion


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--r3_constraint_seed", type=int, required=True)
    parser.add_argument("--r3_pilot_count", type=int, default=8)
    parser.add_argument("--r3_condition_mode", choices=("baseline", "joint", "joint_v2"), required=True)
    opts, remaining = parser.parse_known_args()
    if not 1 <= opts.r3_pilot_count <= 100:
        raise ValueError("pilot_count must be in 1..100")
    original_collate = official.collate_all

    def matched_collate(n_frames: int, dataset: str, task: str):
        if dataset != "humanml" or task != "kick":
            raise ValueError("This controlled pilot only supports HumanML Kick")
        state = np.random.get_state()
        try:
            np.random.seed(opts.r3_constraint_seed)
            texts, hints = original_collate(n_frames, dataset, task)
        finally:
            np.random.set_state(state)
        texts = texts[: opts.r3_pilot_count]
        hints = hints[: opts.r3_pilot_count].copy()
        if opts.r3_condition_mode == "baseline":
            return texts, hints

        norm = Path("dataset/humanml_spatial_norm")
        mean = np.load(norm / "Mean_raw.npy").reshape(22, 3)
        std = np.load(norm / "Std_raw.npy").reshape(22, 3)
        hints = hints.reshape(-1, n_frames, 22, 3)

        def assign(joint: int, frames: np.ndarray, position: tuple[float, float, float]) -> None:
            if np.any(frames >= n_frames):
                raise ValueError("Condition frame exceeds sequence length")
            coordinate = np.asarray(position, dtype=np.float32)
            hints[:, frames, joint] = (coordinate - mean[joint]) / std[joint]

        # HumanML coordinates are x=lateral, y=up, z=forward. The contact and
        # balance signals below are *hypothesized constraints*, not GT labels.
        if opts.r3_condition_mode == "joint_v2":
            # Sparse events keep the number of guided frames comparable with
            # the original 23-frame kick target. The sampler below separately
            # regularizes inter-keyframe contact and temporal smoothness.
            assign(10, np.array([17, 25, 30, 35, 40]), (-0.12, 0.06, 0.00))
            assign(0, np.array([30]), (0.00, 0.93, 0.00))
        else:
            assign(10, np.arange(0, 71), (-0.12, 0.06, 0.00))  # left stance foot
            assign(0, np.array([0, 17, 30, 40, 65]), (0.00, 0.93, 0.00))  # pelvis
        # The official Kick retargeter interprets any right-wrist hint as a
        # separate pick/table task. Avoid joint 21 so both arms truly use the
        # same unchanged retargeter behavior.
        upper_frames = np.array([30]) if opts.r3_condition_mode == "joint_v2" else np.array([17, 30, 40])
        assign(15, upper_frames, (0.00, 1.60, 0.04))  # upright head
        assign(20, upper_frames, (-0.28, 1.20, 0.16))  # left guard
        recovery_frames = np.array([65]) if opts.r3_condition_mode == "joint_v2" else np.array([0, 65])
        assign(11, recovery_frames, (0.12, 0.08, 0.00))  # recovery timing
        return texts, hints.reshape(len(texts), n_frames, -1)

    def inference_loader(**kwargs):
        kwargs["hml_mode"] = "text_only"
        return get_dataset_loader(**kwargs)

    official.collate_all = matched_collate
    official.get_dataset_loader = inference_loader
    if opts.r3_condition_mode == "joint_v2":
        # Diffusion-time guidance operates on human motion before the original
        # G1 retargeter. This is not robot-joint IK or post-hoc path editing.
        def joint_gradients(self, x, hint, mask_hint, joint_ids=None):
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                native = x.permute(0, 3, 2, 1).contiguous().squeeze(2)
                native = native * self.std + self.mean
                positions = recover_from_ric(native, 22)
                spatial = torch.linalg.norm((positions - hint) * mask_hint, dim=-1)
                weights = torch.ones(22, device=x.device, dtype=x.dtype)
                weights[11] = 4.0  # preserve the original right-foot target
                weights[10] = 1.0
                weights[0] = 0.5
                weights[15] = 0.5
                weights[20] = 0.5
                spatial_loss = (spatial * weights[None, None, :]).sum()
                left_velocity = positions[:, 18:40, 10] - positions[:, 17:39, 10]
                contact_proxy = torch.linalg.norm(left_velocity, dim=-1).sum()
                acceleration = positions[:, 2:] - 2 * positions[:, 1:-1] + positions[:, :-2]
                temporal_proxy = torch.linalg.norm(acceleration, dim=-1).sum()
                loss = spatial_loss + 2.0 * contact_proxy + 0.1 * temporal_proxy
                grad = torch.autograd.grad(loss, x)[0]
                grad[..., 0] = 0  # retain official root-motion guidance convention
            return spatial.detach(), grad

        GaussianDiffusion.gradients = joint_gradients
    sys.argv = [sys.argv[0], *remaining]
    official.main()


if __name__ == "__main__":
    main()

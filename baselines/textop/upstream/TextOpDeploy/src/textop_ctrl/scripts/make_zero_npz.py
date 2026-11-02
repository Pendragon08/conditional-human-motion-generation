#!/usr/bin/env python3

import numpy as np
import argparse
import os
import torch
from robotmdar.dtype.motion import get_zero_feature, motion_feature_to_dict


def expand_dof_23_to_29(v: np.ndarray) -> np.ndarray:
    """
    v: shape [T, 23]
    return: shape [T, 29]
    """
    T = v.shape[0]
    out = np.zeros((T, 29), dtype=v.dtype)

    out[:, :19] = v[:, :19]
    out[:, 22:26] = v[:, 19:23]

    return out


# MuJoCo to IsaacLab joint reindexing
mujoco_to_isaaclab_reindex = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24,
    18, 25, 19, 26, 20, 27, 21, 28
]


def create_zero_motion_npz_simple(out: str, num_timesteps: int, fps: int):
    """
    Create a zero motion NPZ file using robotmdar's get_zero_feature directly
    
    Args:
        out: Path to save the NPZ file
        num_timesteps: Number of time steps (T)
        fps: Frames per second
    """

    print(f"Creating zero motion NPZ file using robotmdar get_zero_feature:")
    print(f"  - Output path: {out}")
    print(f"  - Time steps: {num_timesteps}")
    print(f"  - FPS: {fps}")

    # Get zero feature from robotmdar
    print("  - Getting zero feature from robotmdar...")
    zero_feature = get_zero_feature()  # Shape: [57]
    print(f"    Zero feature shape: {zero_feature.shape}")

    # Create motion features for all timesteps
    device = "cpu"
    motion_feat = zero_feature.to(device).reshape(1,
                                                  -1).repeat(num_timesteps, 1)
    motion_dict = motion_feature_to_dict(motion_feat)
    print(f"    Motion feature shape: {motion_feat.shape}")

    # Create zero joint positions [T, 23]
    joint_pos_23 = motion_dict['dof'].numpy()

    # Create zero joint velocities [T, 23]
    joint_vel_23 = np.zeros_like(joint_pos_23)

    # Expand from 23 to 29 DOF
    joint_pos_29_mjc = expand_dof_23_to_29(joint_pos_23)
    joint_vel_29_mjc = expand_dof_23_to_29(joint_vel_23)

    # Apply MuJoCo to IsaacLab reindexing
    joint_pos_29 = joint_pos_29_mjc[:, mujoco_to_isaaclab_reindex]
    joint_vel_29 = joint_vel_29_mjc[:, mujoco_to_isaaclab_reindex]

    body_pos_w = motion_dict['root_trans_offset'].numpy().reshape(-1, 1, 3)
    root_rot = motion_dict['root_rot'].numpy()  # xyzw -> wxyz

    body_quat_w = root_rot[:, [3, 0, 1, 2]].reshape(-1, 1, 4)

    # Save to NPZ file
    np.savez(
        out,
        joint_pos=joint_pos_29,  # Use 29 DOF version with IsaacLab ordering
        joint_vel=joint_vel_29,  # Use 29 DOF version with IsaacLab ordering
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        fps=np.array(fps))

    print(f"Successfully created NPZ file: {out}")
    print(f"Data shapes:")
    print(f"  - joint_pos: {joint_pos_29.shape}")
    print(f"  - joint_vel: {joint_vel_29.shape}")
    print(f"  - body_pos_w: {body_pos_w.shape}")
    print(f"  - body_quat_w: {body_quat_w.shape}")
    print(f"  - fps: {fps}")
    print(f"  - Applied MuJoCo to IsaacLab joint reindexing")


def main():
    parser = argparse.ArgumentParser(
        description=
        'Create zero motion NPZ file using robotmdar get_zero_feature')
    parser.add_argument('--out',
                        default="zero_motion.npz",
                        help='Output NPZ file path')
    parser.add_argument('--timesteps',
                        '-t',
                        type=int,
                        default=100,
                        help='Number of time steps (default: 100)')
    parser.add_argument('--fps',
                        type=int,
                        default=50,
                        help='Frames per second (default: 50)')

    args = parser.parse_args()

    create_zero_motion_npz_simple(out=args.out,
                                  num_timesteps=args.timesteps,
                                  fps=args.fps)


if __name__ == '__main__':
    exit(main())

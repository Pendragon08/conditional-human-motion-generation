"""Produce experimental 263D features for one trusted 36D .npy clip.

The caller must explicitly choose the locked-waist tolerance and contact
threshold. This tool refuses to overwrite an existing artifact. It is a
preprocessing experiment, not the paper's large-scale retargeting solution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from feature263 import encode
from g1_fk import PAPER_22_LINKS, URDFFK
from motion_bridge import prepare_near_locked_clip


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion', type=Path, required=True, help='raw (T,36) npy, 29-DoF source order')
    parser.add_argument('--urdf', type=Path, required=True, help='27-DoF G1 URDF')
    parser.add_argument('--output', type=Path, required=True, help='new .npz output file')
    parser.add_argument('--source-fps', type=int, required=True)
    parser.add_argument('--waist-tolerance-rad', type=float, required=True)
    parser.add_argument('--foot-velocity-squared-threshold', type=float, required=True)
    args = parser.parse_args()
    if args.output.suffix != '.npz':
        parser.error('--output must end with .npz')
    model = URDFFK(args.urdf)
    motion = np.load(args.motion, allow_pickle=False)
    clip = prepare_near_locked_clip(motion, model, source_fps=args.source_fps, target_fps=20,
                                    max_locked_angle_rad=args.waist_tolerance_rad)
    features = encode(clip['points_z_up_world'],
                      foot_velocity_squared_threshold=args.foot_velocity_squared_threshold)
    metadata = {
        'source_sha256': digest(args.motion), 'urdf_sha256': digest(args.urdf),
        'source_fps': args.source_fps, 'target_fps': 20,
        'locked_waist_tolerance_rad': args.waist_tolerance_rad,
        'max_discarded_waist_angle_rad': clip['max_discarded_waist_angle_rad'],
        'foot_velocity_squared_threshold': args.foot_velocity_squared_threshold,
        'encoding': 'experimental HumanML 263D; not validated against authors v2 implementation',
        'keypoints_coordinate_frame': 'world Z-up', 'feature_frame': 'canonical Y-up',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as out:
        np.savez_compressed(out, features=features, root_pos=clip['root_pos'],
                            root_quat_xyzw=clip['root_quat_xyzw'], joint_pos=clip['joint_pos'],
                            joint_names=np.asarray(clip['joint_names']),
                            keypoints_world_z_up=clip['points_z_up_world'],
                            keypoint_links=np.asarray(PAPER_22_LINKS),
                            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(json.dumps({'output': str(args.output), 'feature_shape': list(features.shape), **metadata}, indent=2))


if __name__ == '__main__':
    main()

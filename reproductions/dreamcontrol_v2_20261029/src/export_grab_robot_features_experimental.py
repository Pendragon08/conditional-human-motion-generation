"""Export audited GRAB robot-window witnesses as NON-TRAINING 263D features.

This connects existing 27-DoF window output to FK and the experimental 263D
encoder. It does not establish source FPS, caption alignment, physical quality,
license, or paper equivalence. No training manifest or statistics are emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from feature263 import canonicalize_z_up, encode
from g1_fk import URDFFK
from data_loaders.humanml.scripts.motion_process import recover_from_ric


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def export(artifact, expected_hash, urdf, expected_urdf_hash, output, threshold):
    if sha(artifact) != expected_hash or sha(urdf) != expected_urdf_hash:
        raise ValueError('artifact or URDF differs from audited input')
    with np.load(artifact, allow_pickle=False) as data:
        frames = np.asarray(data['frames20'])
        qpos = np.asarray(data['qpos20'], dtype=np.float64)
        names = tuple(str(x) for x in data['joint_names'])
    if (qpos.ndim != 2 or qpos.shape[1] != 34 or len(qpos) < 3 or
            not np.isfinite(qpos).all() or frames.shape != (len(qpos),) or
            not np.issubdtype(frames.dtype, np.integer) or
            not np.all(np.diff(frames) == 6)):
        raise ValueError('invalid qpos or declared 120-to-20 Hz frame sequence')
    if not np.allclose(np.linalg.norm(qpos[:, :4], axis=1), 1, atol=1e-4, rtol=0):
        raise ValueError('root quaternion must be normalized')
    model = URDFFK(urdf)
    points = model.forward(qpos[:, 4:7], qpos[:, [1, 2, 3, 0]], qpos[:, 7:], list(names))
    features = encode(points, foot_velocity_squared_threshold=threshold)
    with torch.no_grad():
        decoded = recover_from_ric(torch.from_numpy(features), 22).numpy()
    expected = canonicalize_z_up(points)[:-1]
    max_error = float(np.linalg.norm(decoded - expected, axis=-1).max())
    if max_error > 1e-4:
        raise ValueError('263D FK reconstruction mismatch')
    if sha(artifact) != expected_hash or sha(urdf) != expected_urdf_hash:
        raise RuntimeError('input changed during export')
    with Path(output).open('xb') as f:
        np.savez_compressed(f, features=features, source_frames=frames,
                            points_z_up_world=points, decoded_points=decoded)
    return {'input_sha256': expected_hash, 'urdf_sha256': expected_urdf_hash,
            'output_sha256': sha(output), 'source_frames': frames.tolist(),
            'input_count': len(frames), 'feature_rows': len(features),
            'max_reconstruction_error_m': max_error,
            'foot_velocity_squared_threshold': threshold,
            'declared_source_fps': 120, 'nominal_target_fps': 20,
            'training_ready': False,
            'limitations': ['experimental window selection and component FIR',
                'source FPS and floor not independently verified',
                'no source caption pairing or split',
                'retargeting, collision, contact, dynamics and license not accepted',
                'not the unpublished author v2 method']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--artifact-sha256', required=True)
    p.add_argument('--urdf', type=Path, required=True)
    p.add_argument('--urdf-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--foot-threshold', type=float, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    result = export(a.artifact, a.artifact_sha256, a.urdf, a.urdf_sha256,
                    a.output, a.foot_threshold)
    result['export_script_sha256'] = sha(__file__)
    with a.report.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

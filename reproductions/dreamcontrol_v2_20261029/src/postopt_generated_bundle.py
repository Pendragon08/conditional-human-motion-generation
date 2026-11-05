"""Bridge one saved prior sample to an isolated G1 post-optimization bundle.

The caller supplies the scene, coordinate calibration, old solver and an
explicit provenance record. All output remains experimental and unaccepted.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from robot_prior_postopt import postopt_canonical_motion


def _sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def postopt_generated_bundle(*, input_path, input_sha256, output_path, sample_index,
                             solver_provenance, min_free_bytes, **postopt_kwargs):
    """Write a new root/27-joint NPZ, never replacing existing files.

    Input must be a pinned generated-prior NPZ with BxTx22x3 canonical points
    and metadata. Metadata and solver provenance are descriptive, not proof of
    source rights, capture timing, model training or physical validity.
    """
    if (not isinstance(input_sha256, str) or len(input_sha256) != 64 or
        any(c not in '0123456789abcdef' for c in input_sha256)):
        raise ValueError('input SHA-256 must be lower-case hexadecimal')
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError('sample_index must be a nonnegative integer')
    if isinstance(min_free_bytes, bool) or not isinstance(min_free_bytes, int) or min_free_bytes < 0:
        raise ValueError('an explicit nonnegative disk reserve is required')
    if (not isinstance(solver_provenance, dict) or not solver_provenance or
        any(not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in solver_provenance.items())):
        raise ValueError('nonempty solver provenance string mapping is required')
    source = Path(input_path).resolve(strict=True)
    out = Path(output_path)
    if not source.is_file() or _sha(source) != input_sha256:
        raise ValueError('input file or hash mismatch')
    if out.suffix != '.npz' or not out.parent.is_dir():
        raise ValueError('output must be an NPZ in an existing directory')
    if out.exists(): raise FileExistsError('output already exists')
    if shutil.disk_usage(out.parent).free < min_free_bytes: raise RuntimeError('insufficient disk reserve')
    with np.load(source, allow_pickle=False) as data:
        all_points = data['canonical_points']
        if all_points.ndim != 4 or all_points.shape[2:] != (22, 3) or sample_index >= len(all_points):
            raise ValueError('canonical points or sample index invalid')
        points = all_points[sample_index].copy()
        raw_metadata = data['metadata_utf8']
        if raw_metadata.dtype != np.uint8 or raw_metadata.ndim != 1 or raw_metadata.size > 1000000:
            raise ValueError('input metadata invalid')
        source_metadata = json.loads(raw_metadata.tobytes().decode('utf8'))
    if not isinstance(source_metadata, dict): raise ValueError('input metadata must be an object')
    result = postopt_canonical_motion(points, **postopt_kwargs)
    if _sha(source) != input_sha256: raise RuntimeError('input changed during post-optimization')
    metadata = {'input_sha256': input_sha256, 'sample_index': sample_index,
                'input_metadata': source_metadata, 'solver_provenance': solver_provenance,
                'postopt': result['metadata'], 'training_or_physical_acceptance': False}
    raw = json.dumps(metadata, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf8')
    temp = None
    try:
        if shutil.disk_usage(out.parent).free < min_free_bytes: raise RuntimeError('disk reserve fell during optimization')
        with tempfile.NamedTemporaryFile(prefix='.robot-postopt-', suffix='.tmp', dir=out.parent, delete=False) as f:
            temp = Path(f.name)
            np.savez_compressed(f, root_wxyz_xyz=result['root_wxyz_xyz'], joint_pos=result['joint_pos'],
                joint_names=np.asarray(result['joint_names']), input_frame_indices=result['input_frame_indices'],
                left_foot_low_speed_proxy=result['left_foot_low_speed_proxy'],
                right_foot_low_speed_proxy=result['right_foot_low_speed_proxy'],
                metadata_utf8=np.frombuffer(raw, dtype=np.uint8))
            f.flush(); os.fsync(f.fileno())
        if shutil.disk_usage(out.parent).free < min_free_bytes: raise RuntimeError('disk reserve fell while saving')
        os.link(temp, out)
    finally:
        if temp is not None: temp.unlink(missing_ok=True)
    return {'output_path': str(out), 'output_sha256': _sha(out), 'metadata': metadata,
            'training_or_physical_acceptance': False}

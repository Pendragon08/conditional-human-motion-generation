"""Read-only audit of official OmniRetarget archives; this is not retargeting.

The data publisher documents a WXYZ quaternion, XYZ base position, and 29
joint positions, followed by an optional object pose. The precise joint order
is intentionally not assumed: it must be verified against Drake's plant.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


def audit_archive(path: Path) -> dict:
    checksum = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            checksum.update(chunk)
    counts: Counter[str] = Counter()
    fps_counts: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    errors = []
    durations = []
    quat_error_max = 0.0
    base_z_min = float('inf')
    base_z_max = float('-inf')
    with zipfile.ZipFile(path) as archive:
        members = [x for x in archive.infolist() if not x.is_dir() and x.filename.endswith('.npz')]
        for member in members:
            try:
                # No extraction and no pickle deserialization.
                with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as motion:
                    q = np.asarray(motion['qpos'])
                    fps_raw = np.asarray(motion['fps'])
                    if fps_raw.size != 1:
                        raise ValueError('fps must be scalar')
                    fps = float(fps_raw.reshape(-1)[0])
                    if not np.isfinite(fps) or fps <= 0:
                        raise ValueError('fps must be positive and finite')
                    if q.ndim != 2 or q.shape[0] < 2 or q.shape[1] not in (36, 43):
                        raise ValueError(f'expected (T,36) or (T,43), got {q.shape}')
                    if not np.issubdtype(q.dtype, np.number) or not np.isfinite(q).all():
                        raise ValueError('qpos must contain finite numbers')
                    norms = np.linalg.norm(q[:, :4], axis=1)
                    if np.any(norms < 1e-8):
                        raise ValueError('zero base quaternion')
                    if q.shape[1] == 43 and np.any(np.linalg.norm(q[:, 36:40], axis=1) < 1e-8):
                        raise ValueError('zero object quaternion')
                    quat_error_max = max(quat_error_max, float(np.max(np.abs(norms - 1))))
                    base_z_min = min(base_z_min, float(q[:, 6].min()))
                    base_z_max = max(base_z_max, float(q[:, 6].max()))
                    durations.append(len(q) / fps)
                    fps_counts[str(fps)] += 1
                    shapes[str(q.shape[1])] += 1
                    counts['valid'] += 1
            except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
                errors.append({'member': member.filename, 'error': str(exc)})
    return {'archive': path.name, 'sha256': checksum.hexdigest(), 'members': len(members),
            'valid': counts['valid'], 'errors': errors, 'fps_counts': dict(fps_counts),
            'qpos_width_counts': dict(shapes), 'total_duration_seconds': float(sum(durations)),
            'duration_seconds_min': min(durations, default=None),
            'duration_seconds_max': max(durations, default=None),
            'base_quaternion_norm_max_abs_error': quat_error_max if durations else None,
            'base_z_range': [base_z_min, base_z_max] if durations else None,
            'joint_order_verified': False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('refusing to overwrite an existing report')
    reports = [audit_archive(path) for path in args.archives]
    result = {'schema_version': 1, 'archives': reports,
              'notes': ['Z-up interpretation and joint ordering require independent verification.',
                        'A valid array does not establish retargeting or physical quality.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as dest:
        json.dump(result, dest, indent=2, ensure_ascii=False, allow_nan=False)
        dest.write('\n')
    for item in reports:
        print(f"{item['archive']}: {item['valid']}/{item['members']} valid; {len(item['errors'])} errors")
    return 1 if any(x['errors'] for x in reports) else 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Inspect and normalize exceptional object quaternions without modifying archives."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

from normalize_object_quaternion import normalize_object_quaternions
from omniretarget_bridge import convert_layout
from resample_omniretarget import split_resampled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repairs, refused = [], []
    with zipfile.ZipFile(args.archive) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.endswith('.npz'):
                continue
            payload = archive.read(member)
            try:
                with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                    q, fps = data['qpos'], float(data['fps'])
                if q.shape[1] != 43:
                    continue
                norm = np.linalg.norm(q[:, 36:40], axis=1)
                if np.allclose(norm, 1., atol=1e-4, rtol=0):
                    continue
                fixed, meta = normalize_object_quaternions(q)
                assert np.array_equal(fixed[:, :36], q[:, :36])
                assert np.array_equal(fixed[:, 40:], q[:, 40:])
                # Compare direction of the two 4-vectors without claiming the
                # original data express the intended physical orientation.
                direction = q[:, 36:40] / norm[:, None]
                agreement = np.abs(np.sum(direction * fixed[:, 36:40], axis=1))
                clips, dropped = split_resampled(fixed, fps)
                for clip in clips:
                    convert_layout(clip)
                repairs.append({'member': member.filename, 'source_npz_sha256': hashlib.sha256(payload).hexdigest(),
                                'fps': fps, 'source_frames': len(q), 'segments': len(clips), 'dropped_frames': dropped,
                                'max_quaternion_direction_error': float(np.max(np.abs(1.-agreement))), **meta})
            except (ValueError, KeyError, OSError) as exc:
                refused.append({'member': member.filename, 'error': str(exc)})
    report = {'archive': args.archive.name, 'repairs': repairs, 'refused': refused,
              'note': 'In-memory witness only; originals unchanged and no train-ready data emitted.'}
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(report, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write('\n')
    print(f'validated repairs: {len(repairs)}; refused: {len(refused)}')


if __name__ == '__main__':
    main()

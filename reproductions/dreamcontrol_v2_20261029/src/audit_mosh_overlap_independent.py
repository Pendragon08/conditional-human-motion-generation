"""Independent inspection of saved MoSh windows; no optimization or repair."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(sys.argv[1])
    rows_file = root / 'windows.jsonl'
    rows = [json.loads(line) for line in rows_file.read_text().splitlines()]
    if len(rows) != 253:
        raise ValueError('expected 253 complete windows')
    previous = None
    comparisons = []
    for i, row in enumerate(rows):
        path = root / row['artifact']
        if row['window'] != i or digest(path) != row['artifact_sha256']:
            raise ValueError('window identity/hash mismatch')
        with np.load(path, allow_pickle=False) as f:
            frames = f['sample_indices'].copy()
            angles = f['joint_pos'].astype(np.float64)
            names = f['joint_names'].copy()
        if angles.shape != (7, 27) or len(names) != 27 or not np.isfinite(angles).all():
            raise ValueError('invalid joint data')
        start = min(1 + 4*i, 1008)
        if frames.tolist() != row['sample_indices'] or not np.array_equal(frames, np.arange(start, start + 7)):
            raise ValueError('bad source indices')
        if previous is not None:
            old_frames, old_angles, old_names = previous
            common, a, b = np.intersect1d(old_frames, frames, return_indices=True)
            if len(common) != (4 if i == 252 else 3) or not np.array_equal(names, old_names):
                raise ValueError('bad overlap')
            d = np.abs(old_angles[a] - angles[b])
            frame_i, joint_i = np.unravel_index(np.argmax(d), d.shape)
            value = float(d[frame_i, joint_i])
            if not np.isclose(value, row['overlap']['joint_rad'], rtol=0, atol=1e-6):
                raise ValueError('saved overlap statistic mismatch')
            comparisons.append({'window': i, 'frame': int(common[frame_i]), 'joint': str(names[joint_i]),
                                'max_rad': value, 'previous_rad': float(old_angles[a[frame_i], joint_i]),
                                'current_rad': float(angles[b[frame_i], joint_i])})
        previous = frames, angles, names
    values = np.array([x['max_rad'] for x in comparisons])
    out = {'rows_sha256': digest(rows_file), 'script_sha256': digest(Path(__file__)),
           'windows_verified': len(rows), 'overlaps': len(comparisons),
           'joint_max_quantiles_rad': {str(q): float(np.quantile(values, q)) for q in [0, .5, .95, .99, 1]},
           'overlap_over_1_rad_count': int((values > 1).sum()),
           'largest_overlaps': sorted(comparisons, key=lambda x: x['max_rad'], reverse=True)[:8],
           'cpu_gpu_switch_boundary': comparisons[144],
           'limitations': 'Saved outputs only; no independent solver rerun, physical or training acceptance.',
           'training_ready': False}
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

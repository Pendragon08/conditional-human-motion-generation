"""Create a hash-bound visual review bundle from existing G1 state evidence.

Images are for reviewing text/motion correspondence, not physical or source
acceptance. No reference trajectory or training manifest is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from g1_fk import URDFFK

CHAINS = ((0, 2, 5, 8, 11), (0, 1, 4, 7, 10), (0, 3, 6, 9, 12, 15),
          (9, 14, 17, 19, 21), (9, 13, 16, 18, 20))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checked_file(root, name, expected):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError('expected a relative artifact path')
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
        raise ValueError('state file path or hash mismatch')
    return path


def sample_indices(length, count=6):
    if not isinstance(length, int) or length < count or count < 2:
        raise ValueError('not enough state samples')
    return np.rint(np.linspace(0, length - 1, count)).astype(int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--state-manifest', type=Path, required=True)
    parser.add_argument('--state-root', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--urdf-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True, help='new directory; must not exist')
    args = parser.parse_args()
    manifest, state_manifest = args.manifest.resolve(strict=True), args.state_manifest.resolve(strict=True)
    state_root, urdf = args.state_root.resolve(strict=True), args.urdf.resolve(strict=True)
    manifest_hash, state_hash, urdf_hash = digest(manifest), digest(state_manifest), digest(urdf)
    evidence = json.loads(state_manifest.read_text(encoding='utf8'))
    records = json.loads(manifest.read_text(encoding='utf8'))['records']
    if manifest_hash != evidence['preview_manifest_sha256'] or urdf_hash != args.urdf_sha256:
        raise ValueError('manifest or URDF does not match evidence')
    by_id = {r['sample_id']: r for r in evidence['records']}
    if len(by_id) != len(evidence['records']) or set(by_id) != {r['sample_id'] for r in records}:
        raise ValueError('state evidence must match all manifest IDs exactly')
    args.output.mkdir(parents=False, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import textwrap
    robot = URDFFK(urdf)
    rows = []
    for record in records:
        sample = record['sample_id']
        if not sample or Path(sample).name != sample or sample in ('.', '..'):
            raise ValueError('unsafe sample ID')
        row = by_id[sample]
        if row['feature_sha256'] != record['feature_sha256']:
            raise ValueError('feature identity mismatch')
        path = checked_file(state_root, row['state_path'], row['state_sha256'])
        with np.load(path, allow_pickle=False) as data:
            root, q = data['root_wxyz_xyz'].copy(), data['joint_pos'].copy()
            times = data['sample_times'].copy()
            names = [str(x) for x in data['joint_names']]
        if (root.ndim != 2 or root.shape[1] != 7 or q.shape != (len(root), 27) or
            times.shape != (len(root),) or not np.isfinite(root).all() or
            not np.isfinite(q).all() or not np.isfinite(times).all() or
            not np.allclose(np.diff(times), .05, rtol=0, atol=1e-8)):
            raise ValueError('invalid state or nominal 20 Hz timeline')
        points = robot.forward(root[:, 4:], root[:, [1, 2, 3, 0]], q, names)
        if points.shape != (len(root), 22, 3) or not np.isfinite(points).all():
            raise ValueError('invalid FK output')
        ids = sample_indices(len(points))
        # Remove root translation separately in each snapshot to compare pose;
        # these views intentionally do not display global locomotion.
        centered = points - points[:, :1, :]
        radius = max(float(np.max(np.abs(centered))), .2) * 1.12
        fig, axes = plt.subplots(2, len(ids), figsize=(15, 5.1), constrained_layout=True)
        for col, index in enumerate(ids):
            p = centered[index]
            for row_idx, horizontal in enumerate((0, 1)):
                ax = axes[row_idx, col]
                for chain in CHAINS:
                    ax.plot(p[list(chain), horizontal], p[list(chain), 2], '-o', markersize=2, linewidth=1.5)
                ax.axhline(0, color='#999999', linewidth=.45)
                ax.set(xlim=(-radius, radius), ylim=(-radius, radius), aspect='equal')
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0: ax.set_ylabel('x / z' if row_idx == 0 else 'y / z')
                if row_idx == 0: ax.set_title(f'{index}  |  {times[index]:.2f} s')
        caption = '\n'.join(textwrap.wrap(record['caption'], width=110))
        fig.suptitle(f"{sample} ({record['split']}) | {caption}\n"
                     'Root-centered snapshots; approximate G1 FK; not semantic or physical acceptance', fontsize=10)
        target = args.output / f'{sample}.png'
        if target.exists(): raise FileExistsError(target)
        fig.savefig(target, dpi=120)
        plt.close(fig)
        rows.append({'sample_id': sample, 'split': record['split'], 'caption': record['caption'],
                     'feature_sha256': record['feature_sha256'], 'state_sha256': row['state_sha256'],
                     'frame_indices': ids.tolist(), 'sample_times': times[ids].tolist(),
                     'image': target.name, 'image_sha256': digest(target)})
    if digest(manifest) != manifest_hash or digest(state_manifest) != state_hash or digest(urdf) != urdf_hash:
        raise RuntimeError('inputs changed while rendering')
    result = {'manifest_sha256': manifest_hash, 'state_manifest_sha256': state_hash,
              'urdf_sha256': urdf_hash, 'cases': rows,
              'limitations': ['Root translation removed; plots cannot establish locomotion.',
                              'Uses approximate robot FK and source-provided nominal timing.',
                              'Sparse snapshots cannot prove caption, physical, or training quality.']}
    with (args.output / 'index.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'cases': len(rows), 'training_ready': False}))


if __name__ == '__main__':
    main()

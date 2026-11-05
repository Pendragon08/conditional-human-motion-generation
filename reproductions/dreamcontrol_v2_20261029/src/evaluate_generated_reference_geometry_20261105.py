"""Evaluate all admitted references using the public DreamControl geometric metrics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
UPSTREAM = ROOT.parent / 'conditional-human-motion-generation/third_party/DreamControl/TrajGen'
MANIFEST = ROOT / 'stage3_supported_references_20261104/mixed2000_final_v1/native_motion_manifest.json'
MANIFEST_SHA = '6d35b617780968b45ef9d05c2014df17708d6b45bb7158438a13ec6c49df36cf'
METRICS_SHA = '021108acce91e7a4e887ce9b8ff39a2628d9fbbfc54bbd765bee7e24218c1a77'
TASKS = ('pick', 'pour', 'bimanual_box_lift', 'box_push_pull')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_npz(path, digest):
    assert sha(path) == digest, f'content changed: {path}'
    return np.load(path, allow_pickle=False)


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {'mean': float(a.mean()), 'median': float(np.median(a)),
            'p95': float(np.quantile(a, .95)), 'max': float(a.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    assert sha(MANIFEST) == MANIFEST_SHA
    assert sha(UPSTREAM / 'data_loaders/humanml/utils/metrics.py') == METRICS_SHA
    sys.path.insert(0, str(UPSTREAM))
    from data_loaders.humanml.utils.metrics import calculate_skating_ratio, control_l2

    torch.set_num_threads(2)
    manifest = json.loads(MANIFEST.read_text())
    rows = manifest['records']
    assert Counter(r['task'] for r in rows) == dict.fromkeys(TASKS, 500)
    assert len({(r['task'], r['candidate_id']) for r in rows}) == 2000
    per_clip = []

    for row in rows:
        frames = row['source_frames']
        assert row['source_fps'] == 20
        with checked_npz(row['source_sample'], row['source_sample_sha256']) as raw:
            meta = json.loads(raw['metadata_utf8'].tobytes())
            raw_points = raw['canonical_points'].reshape(frames, 22, 3)
            target = raw['target_points'].reshape(frames, 22, 3)
            mask = raw['target_mask'].reshape(frames, 22)
            assert mask.dtype == np.bool_ and mask.any()
        with checked_npz(row['source'], row['source_sha256']) as final:
            assert float(final['fps']) == 20
            world_points = final['keypoints']
            assert world_points.shape == (frames, 22, 3)
        rotation = np.array(meta['canonical_to_world_rotation'])
        translation = np.array(meta['canonical_to_world_translation'])
        assert np.allclose(rotation.T @ rotation, np.eye(3))
        assert np.allclose(rotation[:, 1], [0., 0., 1.])
        assert np.linalg.det(rotation) > 0
        final_points = (world_points - translation) @ rotation
        assert np.isfinite([raw_points, target, final_points]).all()
        assert meta['candidate_id'] == row['candidate_id']
        wrist_mask = mask.copy()
        wrist_mask[:, :20] = False

        out = {'task': row['task'], 'candidate_id': row['candidate_id'], 'frames': frames,
               'conditioned_points': int(mask.sum()), 'conditioned_wrist_points': int(wrist_mask.sum())}
        for name, points in (('raw_diffusion', raw_points), ('final_reference', final_points)):
            error = control_l2(points[None], target[None], mask[None, ..., None])[0]
            assert np.allclose(error[mask], np.linalg.norm(points[mask] - target[mask], axis=-1))
            skate, _ = calculate_skating_ratio(torch.from_numpy(points.transpose(1, 2, 0)[None]))
            out[name] = {'control_l2_m': float(error[mask].mean()),
                         'wrist_only_control_l2_m': float(error[wrist_mask].mean()),
                         'skating_ratio': float(skate[0])}
        per_clip.append(out)

    report = {'status': 'generated_reference_geometry_complete', 'manifest_sha256': MANIFEST_SHA,
              'upstream_metrics_sha256': METRICS_SHA, 'reference_count': len(per_clip),
              'control_l2': 'L2 on actual stored non-padding conditioned points in metres, mean within clip then across clips',
              'skating_ratio': 'public calculate_skating_ratio, 20Hz Y-up, full unpadded clip, indices 10/11, near ground threshold .05m, speed threshold .5m/s, 5-frame window',
              'stage': 'raw diffusion output and matched post-optimized 29-body-joint reference, only for the 2000 admitted clips',
              'tasks': {}, 'clips': per_clip}
    for task in (*TASKS, 'all'):
        group = [row for row in per_clip if task == 'all' or row['task'] == task]
        report['tasks'][task] = {'count': len(group), **{
            stage: {key: stats([row[stage][key] for row in group]) for key in group[0][stage]}
            for stage in ('raw_diffusion', 'final_reference')}}
    with (args.output / 'geometry.json').open('x') as file:
        json.dump(report, file, indent=2, allow_nan=False)
    print(json.dumps(report['tasks'], allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
"""Recompute preview statistics directly from copied train features."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'seven_candidate_technical_preview_20261031'
CROP = ROOT / 'mosh_000798_crop_candidate_20261031'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    summary = json.loads((HERE / 'report.json').read_text())
    manifest = HERE / 'experimental_manifest_not_for_training.json'
    stats_path = HERE / 'experimental_stats_not_for_training.npz'
    assert sha(manifest) == summary['manifest_sha256'] and sha(stats_path) == summary['stats_sha256']
    records = json.loads(manifest.read_text())['records']
    old = json.loads((OLD / 'experimental_manifest_not_for_training.json').read_text())['records']
    assert records[:-1] == old and len(records) == 8
    crop_report = json.loads((CROP / 'report.json').read_text())
    last = records[-1]
    assert last['caption'] == crop_report['original_caption_records'][0]['caption']
    assert last['feature_sha256'] == crop_report['output_sha256'] and not last['caption_accepted']
    assert last['source_id'].endswith('/MPI_mosh/00008/misc_poses.npy')
    xlist, points = [], []
    groups = {}
    for r in records:
        path = HERE / r['feature_path']
        assert sha(path) == r['feature_sha256'] and not r.get('caption_accepted', False)
        assert groups.setdefault(r['group_id'], r['split']) == r['split']
        if r['split'] == 'train':
            with np.load(path, allow_pickle=False) as f:
                x = f['features'].copy().astype(np.float64)
            xlist.append(x)
            with torch.no_grad():
                points.append(recover_from_ric(torch.from_numpy(x.astype(np.float32)), 22).numpy().reshape(-1, 66).astype(np.float64))
    x = np.concatenate(xlist)
    p = np.concatenate(points)
    expected = {'feature_mean': x.mean(0), 'feature_std': np.maximum(x.std(0), .001),
                'spatial_mean': p.mean(0), 'spatial_std': np.maximum(p.std(0), .001)}
    errors = {}
    with np.load(stats_path, allow_pickle=False) as f:
        for key in expected:
            errors[key] = float(np.abs(expected[key] - f[key]).max())
            assert errors[key] < 1e-6
    assert len(x) == summary['train_frames'] == 296
    result = {'manifest_sha256': sha(manifest), 'stats_sha256': sha(stats_path), 'records_verified': len(records),
              'train_frames': len(x), 'max_stat_errors': errors, 'declared_groups': len(groups),
              'unaccepted_captions': len(records), 'training_ready': False,
              'limitations': 'Shares feature decoder; declared source grouping, actual FPS, full semantics and physics not verified.'}
    with (HERE / 'independent_audit.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

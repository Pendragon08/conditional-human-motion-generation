"""Create a protected eight-record state and feature preflight, never training."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from training_stats import training_stats
from data_loaders.humanml.scripts.motion_process import recover_from_ric

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE = ROOT / 'bounded_joint_projection_20261031'
EIGHT = ROOT / 'eight_candidate_technical_preview_20261031'
NEWSTATE = ROOT / 'mosh_000798_state_preflight_20261031'
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def copy_checked(src, dst, expected):
    if sha(src) != expected:
        raise ValueError('input hash mismatch')
    with src.open('rb') as a, dst.open('xb') as b:
        shutil.copyfileobj(a, b)
    if sha(dst) != expected:
        raise ValueError('copy hash mismatch')


def main():
    torch.set_num_threads(1)
    old = BASE / 'output'
    old_report = json.loads((old / 'report.json').read_text())
    old_audit = json.loads((BASE / 'independent_audit.json').read_text())
    if (sha(old / 'report.json') != old_audit['report_sha256'] or
        sha(old / 'manifest.json') != old_report['manifest_sha256'] or
        sha(old / 'state_manifest.json') != old_report['state_manifest_sha256']):
        raise ValueError('projected baseline provenance changed')
    records = json.loads((old / 'manifest.json').read_text())['records']
    states = json.loads((old / 'state_manifest.json').read_text())['records']
    new_report = json.loads((NEWSTATE / 'report.json').read_text())
    new = json.loads((EIGHT / 'experimental_manifest_not_for_training.json').read_text())['records'][-1]
    check = json.loads((EIGHT / 'independent_audit.json').read_text())
    if (sha(EIGHT / 'experimental_manifest_not_for_training.json') != check['manifest_sha256'] or
        new['sample_id'] != 'MOSH_000798_crop' or not new_report['state_technical_checks_passed'] or
        new_report['cases'][0]['feature_sha256'] != new['feature_sha256']):
        raise ValueError('new preview/state evidence mismatch')
    records.append(new)
    states.append({'sample_id': new['sample_id'], 'feature_sha256': new['feature_sha256'],
                   'state_path': 'MOSH_000798_crop_state.npz',
                   'state_sha256': new_report['cases'][0]['state_sha256'],
                   'foot_velocity_squared_threshold': .002})
    index = {x['sample_id']: x for x in states}
    if len(index) != len(records) or set(index) != {x['sample_id'] for x in records}:
        raise ValueError('state records do not match feature records')
    train, points = [], []
    for r in records:
        state = index[r['sample_id']]
        if state['feature_sha256'] != r['feature_sha256']:
            raise ValueError('feature/state mismatch')
        is_new = r['sample_id'] == 'MOSH_000798_crop'
        src = EIGHT / r['feature_path'] if is_new else old / r['feature_path']
        state_src = NEWSTATE / 'state.npz' if is_new else old / state['state_path']
        copy_checked(src, HERE / r['feature_path'], r['feature_sha256'])
        copy_checked(state_src, HERE / state['state_path'], state['state_sha256'])
        if r['split'] == 'train':
            with np.load(src, allow_pickle=False) as f:
                x = f['features'].copy()
            train.append(('train', x))
            with torch.no_grad():
                points.append(recover_from_ric(torch.from_numpy(x), 22).numpy().reshape(-1, 66).astype(np.float64))
    stats = training_stats(train, min_std=.001)
    points = np.concatenate(points)
    with (HERE / 'stats.npz').open('xb') as f:
        np.savez_compressed(f, feature_mean=stats['mean'], feature_std=stats['std'],
                            spatial_mean=points.mean(0), spatial_std=np.maximum(points.std(0), .001))
    for name, value in [('manifest.json', {'records': records, 'training_ready': False}),
                        ('state_manifest.json', {'records': states, 'training_ready': False})]:
        with (HERE / name).open('x', encoding='utf8') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
    cfg = json.loads((BASE / 'preflight_config.json').read_text())
    cfg.update({'dataset_root': str(HERE), 'manifest': str(HERE / 'manifest.json'), 'stats': str(HERE / 'stats.npz'),
                'state_root': str(HERE), 'state_manifest': str(HERE / 'state_manifest.json'),
                'run_dir': str(HERE), 'device': 'cpu', 'purpose': 'unaccepted eight-case CPU technical preflight'})
    with (HERE / 'preflight_config.json').open('x', encoding='utf8') as f:
        json.dump(cfg, f, indent=2, allow_nan=False)
        f.write('\n')
    receipt = {'manifest_sha256': sha(HERE / 'manifest.json'), 'state_manifest_sha256': sha(HERE / 'state_manifest.json'),
               'stats_sha256': sha(HERE / 'stats.npz'), 'base_report_sha256': sha(old / 'report.json'),
               'new_state_report_sha256': sha(NEWSTATE / 'report.json'), 'train_frames': stats['frame_count'],
               'training_ready': False, 'limitations': ['Prior projected states are bounded approximations.',
               'Captions, source identity, real FPS, physical quality and licenses remain unaccepted.']}
    with (HERE / 'build_report.json').open('x', encoding='utf8') as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

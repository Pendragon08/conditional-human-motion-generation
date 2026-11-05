"""Create an unaccepted preview with non-hand model contacts quarantined.

This is a project-specific conservative review aid, not an author filter.
Retained hand contacts and the assumed ground remain unresolved.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from training_stats import training_stats

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT = ROOT / 'eight_projected_preflight_20261031'
CONTACT = ROOT / 'eight_self_contact_classification_20261031'
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def copy(src, dest, expected):
    if sha(src) != expected:
        raise ValueError('source changed')
    with src.open('rb') as f, dest.open('xb') as g:
        shutil.copyfileobj(f, g)
    if sha(dest) != expected:
        raise ValueError('copy changed')


def main():
    torch.set_num_threads(1)
    classification = json.loads((CONTACT / 'report.json').read_text())
    check = json.loads((CONTACT / 'audit_report.json').read_text())
    if sha(CONTACT / 'report.json') != check['report_sha256']:
        raise ValueError('contact report changed')
    parent = json.loads((INPUT / 'build_report.json').read_text())
    if (sha(INPUT / 'manifest.json') != parent['manifest_sha256'] or
        sha(INPUT / 'state_manifest.json') != parent['state_manifest_sha256']):
        raise ValueError('parent manifests changed')
    records = json.loads((INPUT / 'manifest.json').read_text())['records']
    states = {x['sample_id']: x for x in json.loads((INPUT / 'state_manifest.json').read_text())['records']}
    observed = {x['sample_id']: x for x in classification['cases']}
    if len(records) != 8 or set(states) != set(observed):
        raise ValueError('input coverage changed')
    keep, kept_states, quarantine, train, spatial = [], [], [], [], []
    for r in records:
        sample = r['sample_id']; original = observed[sample]
        if original['other_contact_frames']:
            quarantine.append({'sample_id': sample, 'other_contact_frames': original['other_contact_frames'],
                               'reason': 'non-hand self-penetration in a specific MuJoCo asset; requires review'})
            continue
        state = states[sample]
        if sha(INPUT / r['feature_path']) != r['feature_sha256'] or sha(INPUT / state['state_path']) != state['state_sha256']:
            raise ValueError('source hash changed')
        new = dict(r)
        if sample == 'ACCAD_004347':
            new['split'] = 'val'
            new['auxiliary_split_note'] = 'calibration action reserved as conservative source-group holdout; official split unchanged'
        copy(INPUT / r['feature_path'], HERE / r['feature_path'], r['feature_sha256'])
        copy(INPUT / state['state_path'], HERE / state['state_path'], state['state_sha256'])
        keep.append(new); kept_states.append(state)
        if new['split'] == 'train':
            with np.load(HERE / new['feature_path'], allow_pickle=False) as f:
                x = f['features'].copy()
            train.append(('train', x))
            with torch.no_grad():
                spatial.append(recover_from_ric(torch.from_numpy(x), 22).numpy().reshape(-1, 66).astype(np.float64))
    if len(keep) != 5 or len(quarantine) != 3:
        raise ValueError('unexpected conditional triage')
    stats = training_stats(train, min_std=.001); spatial = np.concatenate(spatial)
    with (HERE / 'stats.npz').open('xb') as f:
        np.savez_compressed(f, feature_mean=stats['mean'], feature_std=stats['std'],
                            spatial_mean=spatial.mean(0), spatial_std=np.maximum(spatial.std(0), .001))
    for name, content in [('manifest.json', {'records': keep, 'training_ready': False}),
                          ('state_manifest.json', {'records': kept_states, 'training_ready': False})]:
        with (HERE / name).open('x', encoding='utf8') as f:
            json.dump(content, f, indent=2, allow_nan=False); f.write('\n')
    config = json.loads((INPUT / 'preflight_config.json').read_text())
    config.update({'dataset_root': str(HERE), 'manifest': str(HERE / 'manifest.json'), 'stats': str(HERE / 'stats.npz'),
                   'state_root': str(HERE), 'state_manifest': str(HERE / 'state_manifest.json'),
                   'run_dir': str(HERE), 'device': 'cpu', 'purpose': 'unaccepted 5-record diagnostic triage'})
    with (HERE / 'preflight_config.json').open('x', encoding='utf8') as f:
        json.dump(config, f, indent=2, allow_nan=False); f.write('\n')
    result = {'parent_manifest_sha256': sha(INPUT / 'manifest.json'),
              'classification_sha256': sha(CONTACT / 'report.json'), 'script_sha256': sha(Path(__file__)),
              'manifest_sha256': sha(HERE / 'manifest.json'), 'state_manifest_sha256': sha(HERE / 'state_manifest.json'),
              'stats_sha256': sha(HERE / 'stats.npz'), 'kept_ids': [r['sample_id'] for r in keep],
              'quarantined': quarantine, 'train_frames': stats['frame_count'], 'training_ready': False,
              'limitations': ['Project-specific conservative triage, not the paper filter or proof of a defect.',
                              'Retained hand contacts, ground, source identity, caption semantics and dynamics unresolved.',
                              'Validation is one stationary calibration motion; this tiny preview is not sufficient evidence of a robust model.']}
    with (HERE / 'report.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'retained': len(keep), 'quarantined': len(quarantine), 'train_frames': stats['frame_count'], 'training_ready': False}))


if __name__ == '__main__':
    main()

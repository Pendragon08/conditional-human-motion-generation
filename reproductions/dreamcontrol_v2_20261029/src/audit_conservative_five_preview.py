"""Recheck quarantine partition, copied bytes and train-only feature moments."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT = ROOT / 'eight_projected_preflight_20261031'
CONTACT = ROOT / 'eight_self_contact_classification_20261031'
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    report = json.loads((HERE / 'report.json').read_text())
    if sha(INPUT / 'manifest.json') != report['parent_manifest_sha256'] or sha(CONTACT / 'report.json') != report['classification_sha256']:
        raise ValueError('evidence changed')
    for key, filename in [('manifest_sha256','manifest.json'),('state_manifest_sha256','state_manifest.json'),('stats_sha256','stats.npz')]:
        if sha(HERE / filename) != report[key]: raise ValueError('output changed')
    prior = {r['sample_id']:r for r in json.loads((INPUT/'manifest.json').read_text())['records']}
    before_states = {r['sample_id']:r for r in json.loads((INPUT/'state_manifest.json').read_text())['records']}
    obs = {r['sample_id']:r for r in json.loads((CONTACT/'report.json').read_text())['cases']}
    records = json.loads((HERE/'manifest.json').read_text())['records']
    states = {r['sample_id']:r for r in json.loads((HERE/'state_manifest.json').read_text())['records']}
    kept = {r['sample_id'] for r in records}; excluded = {x['sample_id'] for x in report['quarantined']}
    if kept | excluded != set(prior) or kept & excluded or excluded != {x for x in obs if obs[x]['other_contact_frames'] > 0}:
        raise ValueError('incorrect conservative quarantine')
    groups = {}; splits = {}; train = []
    for r in records:
        name = r['sample_id']; orig = prior[name]; state = states[name]
        expected_split = 'val' if name == 'ACCAD_004347' else orig['split']
        if (r['split'] != expected_split or r['feature_sha256'] != orig['feature_sha256'] or
            state != before_states[name] or sha(HERE / r['feature_path']) != r['feature_sha256'] or
            sha(HERE / state['state_path']) != state['state_sha256'] or r.get('caption_accepted')):
            raise ValueError('retained record mismatch')
        if groups.setdefault(r['group_id'], r['split']) != r['split']:
            raise ValueError('group crosses splits')
        splits[r['split']] = splits.get(r['split'], 0) + 1
        if r['split'] == 'train':
            with np.load(HERE / r['feature_path'], allow_pickle=False) as f:
                train.append(f['features'].astype(float))
    data = np.concatenate(train)
    with np.load(HERE/'stats.npz', allow_pickle=False) as f:
        mean_err = float(np.abs(f['feature_mean']-data.mean(0)).max())
        std_err = float(np.abs(f['feature_std']-np.maximum(data.std(0),.001)).max())
    if max(mean_err, std_err) > 1e-6 or len(data) != report['train_frames']:
        raise ValueError('train statistics mismatch')
    actual = next(json.loads(line) for line in (HERE/'preflight.log').read_text().splitlines() if line.startswith('{'))
    if (actual['manifest_sha256'] != report['manifest_sha256'] or not actual['state_audit']['state_technical_checks_passed'] or
        actual['feature_audit']['training_ready'] or actual['feature_audit']['caption_status_counts']['unaccepted_or_unspecified'] != len(records)):
        raise ValueError('preflight mismatch')
    if 'training blocked: source and applicable quality acceptance not yet implemented' not in (HERE/'run_rejected.log').read_text():
        raise ValueError('training guard did not reject')
    out = {'report_sha256': sha(HERE/'report.json'), 'retained': sorted(kept), 'quarantined': sorted(excluded),
           'split_counts': splits, 'declared_groups': len(groups), 'train_frames': len(data),
           'train_feature_mean_max_error': mean_err, 'train_feature_std_max_error': std_err,
           'unaccepted_captions': len(records), 'training_ready': False,
           'limitations': 'Did not independently validate contacts, caption semantics, spatial statistics, FK, true ground or real physics.'}
    with (HERE/'independent_audit.json').open('x', encoding='utf8') as f:
        json.dump(out, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()

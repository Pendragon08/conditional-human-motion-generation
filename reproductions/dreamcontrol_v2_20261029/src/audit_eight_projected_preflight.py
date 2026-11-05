"""Recheck eight-case state evidence and train-only feature statistics."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    built = json.loads((HERE / 'build_report.json').read_text())
    manifest, state_manifest, stats = (HERE / x for x in ('manifest.json', 'state_manifest.json', 'stats.npz'))
    assert sha(manifest) == built['manifest_sha256']
    assert sha(state_manifest) == built['state_manifest_sha256']
    assert sha(stats) == built['stats_sha256']
    records = json.loads(manifest.read_text())['records']
    states = {x['sample_id']: x for x in json.loads(state_manifest.read_text())['records']}
    assert len(records) == len(states) == 8
    arrays = []
    for r in records:
        s = states[r['sample_id']]
        assert s['feature_sha256'] == r['feature_sha256']
        assert sha(HERE / r['feature_path']) == r['feature_sha256']
        assert sha(HERE / s['state_path']) == s['state_sha256']
        assert not r.get('caption_accepted', False)
        with np.load(HERE / r['feature_path'], allow_pickle=False) as f:
            feature = f['features'].astype(np.float64)
        with np.load(HERE / s['state_path'], allow_pickle=False) as f:
            assert len(f['joint_pos']) == len(feature) + 1
        if r['split'] == 'train':
            arrays.append(feature)
    data = np.concatenate(arrays)
    with np.load(stats, allow_pickle=False) as f:
        mean_diff = float(np.abs(f['feature_mean'] - data.mean(0)).max())
        std_diff = float(np.abs(f['feature_std'] - np.maximum(data.std(0), .001)).max())
    assert len(data) == 296 and max(mean_diff, std_diff) < 1e-6
    logged = next(json.loads(x) for x in (HERE / 'preflight.log').read_text().splitlines() if x.startswith('{'))
    assert logged['manifest_sha256'] == sha(manifest) and logged['stats_sha256'] == sha(stats)
    assert logged['state_audit']['state_technical_checks_passed'] and len(logged['state_audit']['cases']) == 8
    assert not logged['feature_audit']['training_ready'] and logged['feature_audit']['caption_status_counts']['unaccepted_or_unspecified'] == 8
    assert 'training blocked: source and applicable quality acceptance not yet implemented' in (HERE / 'run_rejected.log').read_text()
    result = {'manifest_sha256': sha(manifest), 'state_manifest_sha256': sha(state_manifest), 'stats_sha256': sha(stats),
              'preflight_log_sha256': sha(HERE / 'preflight.log'), 'run_rejection_log_sha256': sha(HERE / 'run_rejected.log'),
              'records_verified': 8, 'train_frames': len(data), 'feature_mean_max_error': mean_diff,
              'feature_std_max_error': std_diff, 'states_technical_passed': 8, 'captions_unaccepted': 8,
              'training_ready': False, 'limitations': 'Does not independently recompute spatial statistics, FK, encoding, text semantics or physics.'}
    with (HERE / 'independent_audit.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

"""Independent scalar joint/metadata and train-stat audit; no FK replay."""
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for key in ('old_manifest', 'old_features', 'old_states', 'old_state_manifest', 'urdf', 'new_root', 'output'):
        p.add_argument('--' + key.replace('_', '-'), required=True, type=Path)
    a = p.parse_args()
    report = json.loads((a.new_root / 'report.json').read_text())
    old = json.loads(a.old_manifest.read_text())['records']
    oldstate = {r['sample_id']: r for r in json.loads(a.old_state_manifest.read_text())['records']}
    new = {r['sample_id']: r for r in json.loads((a.new_root / 'manifest.json').read_text())['records']}
    newstate = {r['sample_id']: r for r in json.loads((a.new_root / 'state_manifest.json').read_text())['records']}
    measures = {r['sample_id']: r for r in report['cases']}
    assert sha(a.old_manifest) == report['source_manifest_sha256']
    assert sha(a.old_state_manifest) == report['source_state_manifest_sha256']
    assert sha(a.urdf) == report['urdf_sha256']
    for key, file in [('manifest_sha256', 'manifest.json'), ('state_manifest_sha256', 'state_manifest.json'), ('stats_sha256', 'stats.npz')]:
        assert report[key] == sha(a.new_root / file)
    assert len(old) == len(oldstate) == len(new) == len(newstate) == len(measures)
    assert {r['sample_id'] for r in old} == set(oldstate) == set(new) == set(newstate) == set(measures)
    limits = {}
    for joint in ET.parse(a.urdf).getroot().findall('joint'):
        if joint.get('type') == 'revolute':
            lim = joint.find('limit'); limits[joint.get('name')] = (float(lim.get('lower')), float(lim.get('upper')))
        elif joint.get('type') == 'continuous': limits[joint.get('name')] = (-float('inf'), float('inf'))
    assert len(limits) == 27
    trains = []; changed = 0
    for rec in old:
        name = rec['sample_id']; old_s = oldstate[name]; new_s = newstate[name]; new_r = new[name]; m = measures[name]
        assert rec['caption'] == new_r['caption'] and rec['split'] == new_r['split']
        assert new_r['caption_accepted'] is False and new_r['training_ready'] is False
        assert rec['feature_sha256'] == old_s['feature_sha256'] == new_r['parent_feature_sha256']
        assert sha(a.old_features / rec['feature_path']) == rec['feature_sha256']
        assert sha(a.old_states / old_s['state_path']) == old_s['state_sha256'] == new_s['parent_state_sha256']
        assert sha(a.new_root / new_s['state_path']) == new_s['state_sha256']
        assert sha(a.new_root / new_r['feature_path']) == new_r['feature_sha256'] == new_s['feature_sha256']
        with np.load(a.old_states / old_s['state_path'], allow_pickle=False) as before, np.load(a.new_root / new_s['state_path'], allow_pickle=False) as after:
            for field in ('root_wxyz_xyz', 'sample_times', 'joint_names'):
                assert np.array_equal(before[field], after[field])
            q, q2 = before['joint_pos'], after['joint_pos']; names = [str(x) for x in before['joint_names']]
            count = 0; max_delta = 0.
            for i in range(len(q)):
                for j, joint in enumerate(names):
                    lo, hi = limits[joint]
                    expected = min(max(float(q[i, j]), lo), hi)
                    assert q2[i, j] == expected
                    delta = abs(expected - float(q[i, j])); max_delta = max(max_delta, delta)
                    count += int(delta > 0)
            assert count == m['joint_values_changed'] and max_delta == m['max_joint_correction_rad']
            assert max_delta <= report['max_correction_rad']
            assert m['after']['joint_limits_passed'] is True
            changed += int(count > 0)
        with np.load(a.old_features / rec['feature_path'], allow_pickle=False) as old_f, np.load(a.new_root / new_r['feature_path'], allow_pickle=False) as new_f:
            old_x, new_x = old_f['features'], new_f['features']
            assert old_x.shape == new_x.shape and np.isfinite(new_x).all()
            assert int(np.count_nonzero(old_x[:, -4:] != new_x[:, -4:])) == m['foot_proxy_values_changed']
            assert float(np.max(np.abs(old_x.astype(np.float64) - new_x.astype(np.float64)))) == m['max_feature_difference']
            if rec['split'] == 'train': trains.append(new_x.astype(np.float64))
    train = np.concatenate(trains)
    with np.load(a.new_root / 'stats.npz', allow_pickle=False) as stats:
        assert np.allclose(stats['feature_mean'], train.mean(axis=0), rtol=0, atol=1e-12)
        assert np.allclose(stats['feature_std'], np.maximum(train.std(axis=0), .001), rtol=0, atol=1e-12)
    assert len(train) == report['train_frames']
    result = {'cases': len(old), 'changed_cases': changed, 'train_frames': len(train),
              'report_sha256': sha(a.new_root / 'report.json'), 'scalar_projection_and_feature_stats_passed': True,
              'training_ready': False,
              'limitations': ['Did not independently recompute FK, encoding or spatial statistics.',
                              'No caption, actual source time, source identity, contact or physical acceptance.']}
    with a.output.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__': main()

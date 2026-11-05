"""Recompute seven-record preview statistics and validate its source links."""
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'six_candidate_technical_preview_20261031'
CAND = ROOT / 'accad_004347_fullrate_candidate_20261031'
HUMAN = ROOT / 'accad_004347_fullrate_20261031'
TEXTS = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    torch.set_num_threads(1)
    report = json.loads((HERE / 'report.json').read_text())
    manifest_path = HERE / 'experimental_manifest_not_for_training.json'
    stats_path = HERE / 'experimental_stats_not_for_training.npz'
    assert sha(manifest_path) == report['manifest_sha256'] and sha(stats_path) == report['stats_sha256']
    old_path = OLD / 'experimental_manifest_not_for_training.json'
    assert sha(old_path) == report['old_manifest_sha256']
    old = {r['sample_id']: r for r in json.loads(old_path.read_text())['records']}
    manifest = json.loads(manifest_path.read_text())
    candidate = json.loads((CAND / 'report.json').read_text())
    candidate_audit = json.loads((CAND / 'audit_report.json').read_text())
    human = json.loads((HUMAN / 'human_report.json').read_text())
    assert sha(CAND / 'report.json') == report['candidate_report_sha256'] == candidate_audit['candidate_report_sha256']
    assert sha(HUMAN / 'human_report.json') == report['human_report_sha256']
    counts, train, points, groups = Counter(), [], [], {}
    with zipfile.ZipFile(TEXTS) as archive:
        text_bytes = archive.read('texts/004347.txt')
    assert hashlib.sha256(text_bytes).hexdigest() == human['caption_member_sha256']
    original_caption = text_bytes.decode('utf8').splitlines()[0].split('#')[0]
    for rec in manifest['records']:
        sid = rec['sample_id']
        assert rec['caption_accepted'] is False and rec['training_ready'] is False
        for namespace, values in rec['provenance_keys'].items():
            for value in values:
                assert groups.setdefault((namespace, value), rec['split']) == rec['split']
        path = HERE / rec['feature_path']
        assert sha(path) == rec['feature_sha256']
        if sid in old:
            assert rec == old[sid] and sha(OLD / rec['feature_path']) == rec['feature_sha256']
        else:
            assert sid == 'ACCAD_004347' and rec['split'] == 'train'
            assert rec['caption'] == original_caption and rec['caption'] in human['captions']
            assert rec['source_id'] == human['source_path'] and rec['provenance_keys']['potential_subject'] == ['ACCAD/Male1']
            assert sha(CAND / 'candidate.npz') == candidate['output_sha256'] == candidate_audit['artifact_sha256'] == rec['feature_sha256']
        counts[rec['split']] += 1
        with np.load(path, allow_pickle=False) as item:
            features = item['features'].copy()
        if rec['split'] == 'train':
            train.append(features.astype(np.float64))
            with torch.no_grad():
                points.append(recover_from_ric(torch.from_numpy(features), 22).numpy().reshape(-1, 66).astype(np.float64))
    features, points = np.concatenate(train), np.concatenate(points)
    expected = {'feature_mean': features.mean(0), 'feature_std': np.maximum(features.std(0), .001),
                'spatial_mean': points.mean(0), 'spatial_std': np.maximum(points.std(0), .001)}
    with np.load(stats_path, allow_pickle=False) as item:
        errors = {key: float(np.max(np.abs(item[key] - val))) for key, val in expected.items()}
    assert max(errors.values()) < 1e-6 and len(features) == report['train_frames'] == 148
    assert counts == Counter({'train': 3, 'val': 1, 'test': 3})
    result = {'status': 'passed', 'manifest_sha256': sha(manifest_path), 'stats_sha256': sha(stats_path),
              'counts': dict(counts), 'train_frames': len(features), 'statistics_max_abs_errors': errors,
              'accad_caption_member_sha256': human['caption_member_sha256'], 'training_ready': False,
              'limitations': ['Shares joint decoder; no independent FK or encoding.',
                              'Caption trace is not semantic validation; no physical acceptance.',
                              'Source grouping is limited to declared keys; the Male1 relation is conservative.']}
    with (HERE / 'independent_audit.json').open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

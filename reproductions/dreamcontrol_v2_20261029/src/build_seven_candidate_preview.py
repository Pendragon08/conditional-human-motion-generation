"""Extend the isolated six-record data interface with ACCAD 004347.

This produces only an unaccepted technical preview. It does not decide whether
the source, time, caption, model, retargeting or physics is fit for training.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from audit_robot_features import audit
from training_stats import training_stats
from data_loaders.humanml.scripts.motion_process import recover_from_ric

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'six_candidate_technical_preview_20261031'
CAND = ROOT / 'accad_004347_fullrate_candidate_20261031'
HUMAN = ROOT / 'accad_004347_fullrate_20261031'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    torch.set_num_threads(1)
    old_manifest = OLD / 'experimental_manifest_not_for_training.json'
    old_audit = json.loads((OLD / 'independent_audit.json').read_text())
    if sha(old_manifest) != old_audit['manifest_sha256']:
        raise ValueError('old preview manifest changed')
    records = json.loads(old_manifest.read_text())['records']
    report_path = CAND / 'report.json'
    candidate = json.loads(report_path.read_text())
    candidate_audit = json.loads((CAND / 'audit_report.json').read_text())
    human_path = HUMAN / 'human_report.json'
    human = json.loads(human_path.read_text())
    human_audit = json.loads((HUMAN / 'human_audit.json').read_text())
    if (sha(report_path) != candidate_audit['candidate_report_sha256'] or
        sha(CAND / 'candidate.npz') != candidate_audit['artifact_sha256'] or
        sha(CAND / 'candidate.npz') != candidate['output_sha256'] or
        sha(human_path) != candidate['human_report_sha256'] or
        sha(human_path) != human_audit['human_report_sha256'] or
        human['source_sha256'] != human_audit['source_sha256'] or
        candidate['source_sha256'] != human['source_sha256'] or
        candidate['captions'] != human['captions']):
        raise ValueError('candidate provenance or captions changed')
    records.append({'sample_id': 'ACCAD_004347', 'source_id': human['source_path'],
                    'group_id': 'conservative:ACCAD/Male1',
                    'provenance_keys': {'source_path': [human['source_path']],
                                        'source_directory': [human['source_parent']],
                                        'potential_subject': ['ACCAD/Male1']},
                    'split': 'train', 'caption': human['captions'][0],
                    'caption_source': 'HumanML texts.zip: texts/004347.txt',
                    'caption_accepted': False, 'feature_path': 'ACCAD_004347.npz',
                    'feature_sha256': candidate['output_sha256'], 'fps': 20,
                    'training_ready': False})
    arrays = []
    for rec in records:
        source = CAND / 'candidate.npz' if rec['sample_id'] == 'ACCAD_004347' else OLD / rec['feature_path']
        if sha(source) != rec['feature_sha256']:
            raise ValueError('source feature changed')
        dest = HERE / rec['feature_path']
        with source.open('rb') as src, dest.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        if sha(dest) != rec['feature_sha256']:
            raise ValueError('copied feature changed')
        with np.load(dest, allow_pickle=False) as data:
            arrays.append((rec['split'], data['features'].copy()))
    stats = training_stats(arrays, min_std=.001)
    points = []
    for split, features in arrays:
        if split == 'train':
            with torch.no_grad():
                points.append(recover_from_ric(torch.from_numpy(features), 22).numpy().reshape(-1, 66).astype(np.float64))
    points = np.concatenate(points)
    values = {'feature_mean': stats['mean'], 'feature_std': stats['std'],
              'spatial_mean': points.mean(0), 'spatial_std': np.maximum(points.std(0), .001)}
    with (HERE / 'experimental_stats_not_for_training.npz').open('xb') as output:
        np.savez_compressed(output, **values)
    manifest = {'records': records, 'training_ready': False, 'purpose': 'unaccepted technical preview',
                'limitations': ['captions, source identity, FPS and physical quality unaccepted',
                                'auxiliary grouping is not the official benchmark split',
                                'ACCAD Male1 is conservatively grouped, not independently identified']}
    with (HERE / 'experimental_manifest_not_for_training.json').open('x', encoding='utf8') as output:
        json.dump(manifest, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write('\n')
    result = audit(HERE, records, values, min_std=.001)
    if result['training_ready'] or not result['explicit_provenance_groups_disjoint']:
        raise ValueError('unexpected audit result')
    summary = {'old_manifest_sha256': sha(old_manifest), 'candidate_report_sha256': sha(report_path),
               'candidate_audit_sha256': sha(CAND / 'audit_report.json'), 'human_report_sha256': sha(human_path),
               'manifest_sha256': sha(HERE / 'experimental_manifest_not_for_training.json'),
               'stats_sha256': sha(HERE / 'experimental_stats_not_for_training.npz'),
               'train_frames': stats['frame_count'], 'clamped_feature_dimensions': len(stats['clamped_channels']),
               'feature_audit': result, 'training_ready': False, 'limitations': manifest['limitations']}
    with (HERE / 'report.json').open('x', encoding='utf8') as output:
        json.dump(summary, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write('\n')
    print(json.dumps({'train_frames': stats['frame_count'], 'counts': result['counts'], 'training_ready': False}))


if __name__ == '__main__':
    main()

"""Add the unaccepted MoSh crop to the source-grouped technical preview."""
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
OLD = ROOT / 'seven_candidate_technical_preview_20261031'
CROP = ROOT / 'mosh_000798_crop_candidate_20261031'
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    old_manifest = OLD / 'experimental_manifest_not_for_training.json'
    old_audit = json.loads((OLD / 'independent_audit.json').read_text())
    if sha(old_manifest) != old_audit['manifest_sha256']:
        raise ValueError('prior manifest changed')
    records = json.loads(old_manifest.read_text())['records']
    report = json.loads((CROP / 'report.json').read_text())
    check = json.loads((CROP / 'audit_report.json').read_text())
    if (sha(CROP / 'report.json') != check['candidate_report_sha256'] or
        sha(CROP / 'candidate.npz') != check['output_sha256'] or
        sha(CROP / 'candidate.npz') != report['output_sha256'] or
        report['caption_accepted'] or report['training_ready']):
        raise ValueError('candidate provenance or status changed')
    if report['official_split'] != 'train' or report['id'] != '000798':
        raise ValueError('unexpected official source')
    # All captions remain recorded in the crop report; the first is used here
    # only to exercise the existing one-caption technical loader.
    records.append({'sample_id': 'MOSH_000798_crop', 'source_id': 'pose_data/MPI_mosh/00008/misc_poses.npy',
                    'group_id': 'conservative:MPI_mosh/00008',
                    'provenance_keys': {'source_path': ['pose_data/MPI_mosh/00008/misc_poses.npy'],
                                        'source_directory': [report['source_group_candidate']],
                                        'potential_subject': ['MPI_mosh/00008']},
                    'split': 'train', 'caption': report['original_caption_records'][0]['caption'],
                    'caption_source': 'HumanML texts.zip: texts/000798.txt; whole segment cropped 1.05 nominal seconds',
                    'caption_accepted': False, 'feature_path': 'MOSH_000798_crop.npz',
                    'feature_sha256': report['output_sha256'], 'fps': 20, 'training_ready': False})
    arrays = []
    for rec in records:
        source = CROP / 'candidate.npz' if rec['sample_id'] == 'MOSH_000798_crop' else OLD / rec['feature_path']
        if sha(source) != rec['feature_sha256']:
            raise ValueError('feature hash changed')
        dest = HERE / rec['feature_path']
        with source.open('rb') as src, dest.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        if sha(dest) != rec['feature_sha256']:
            raise ValueError('feature copy changed')
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
    with (HERE / 'experimental_stats_not_for_training.npz').open('xb') as f:
        np.savez_compressed(f, **values)
    limitations = ['technical preview only; captions, real FPS, source identity and physical quality unaccepted',
                   'auxiliary grouping is not the official benchmark split; subject keys are provisional',
                   'MoSh crop removes 1.05 nominal seconds; whole-segment caption semantics are unverified']
    with (HERE / 'experimental_manifest_not_for_training.json').open('x', encoding='utf8') as f:
        json.dump({'records': records, 'training_ready': False, 'limitations': limitations}, f, indent=2, allow_nan=False)
        f.write('\n')
    checked = audit(HERE, records, values, min_std=.001)
    if checked['training_ready'] or not checked['explicit_provenance_groups_disjoint']:
        raise ValueError('unexpected audit result')
    result = {'old_manifest_sha256': sha(old_manifest), 'candidate_report_sha256': sha(CROP / 'report.json'),
              'candidate_audit_sha256': sha(CROP / 'audit_report.json'),
              'manifest_sha256': sha(HERE / 'experimental_manifest_not_for_training.json'),
              'stats_sha256': sha(HERE / 'experimental_stats_not_for_training.npz'),
              'train_frames': stats['frame_count'], 'clamped_feature_dimensions': len(stats['clamped_channels']),
              'feature_audit': checked, 'training_ready': False, 'limitations': limitations}
    with (HERE / 'report.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'train_frames': stats['frame_count'], 'counts': checked['counts'], 'training_ready': False}))


if __name__ == '__main__':
    main()

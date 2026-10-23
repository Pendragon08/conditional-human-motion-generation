"""Verify every actual episode and write a separate, non-training-ready subset.

Never erase rejected data or turn an external tracker into the paper's GMT.
Only source references, receipts and a training-only normalizer are written.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.phygile_code/'reimplementation'))
    from features import BlockNormalizer
    manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
    summary = json.loads((a.records/'summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'AUXILIARY_PHYSICAL_SCREEN_COMPLETED' or not summary['inputs_unchanged']:
        raise ValueError('A complete immutable physical screen is required')
    if summary['paper_phygile_gmt_filter_reproduced'] or summary['generator_training_allowed_by_this_receipt']:
        raise ValueError('Unexpected external-to-author promotion')
    entries = [(s, e) for s, rows in manifest['splits'].items() for e in rows]
    seeds = summary['seeds']
    if len(summary['records']) != len(entries)*len(seeds):
        raise ValueError('Missing episodes cannot pass filtering')
    by_key = {}
    for row in summary['records']:
        key = row['index'], row['seed']
        if key in by_key or row['seed'] not in seeds or not 0 <= row['index'] < len(entries):
            raise ValueError('Duplicate or foreign episode')
        by_key[key] = row
    expected = {(i, s) for i in range(len(entries)) for s in seeds}
    if set(by_key) != expected:
        raise ValueError('Episode coverage mismatch')
    thresholds = summary['thresholds']
    verified = []
    for (index, seed), row in sorted(by_key.items()):
        split, entry = entries[index]
        if row['features_sha256'] != entry['sha256'] or row['split'] != split:
            raise ValueError('Physical execution and manifest identity differ')
        raw = a.records/f'pair_{index:03d}_seed_{seed}/actual_pre_reset.npz'
        if sha(raw) != row['raw_sha256']:
            raise ValueError('Actual raw record hash mismatch')
        with np.load(raw, allow_pickle=False) as x:
            for name in x.files:
                if name != 'body_names' and not np.isfinite(x[name]).all():
                    raise ValueError('Nonfinite raw record')
            count = len(x['frame'])
            if not np.array_equal(x['frame'], np.arange(count)) or count != row['control_frames']:
                raise ValueError('Missing/dropped/reordered control observations')
            np.testing.assert_allclose(x['time'], (np.arange(count)+1)/50, atol=1e-9, rtol=0)
            if len(x['reference_native']) != row['reference_frames']:
                raise ValueError('Wrong reference horizon')
            body_error = np.linalg.norm(x['body_pos']-x['reference_body_pos'][:count], axis=-1).mean(-1)
            pelvic = np.abs(x['qpos'][:, 2]-x['reference_native'][:count, 2])
            np.testing.assert_allclose(body_error, x['errors'][:, 0], atol=1e-12, rtol=0)
            np.testing.assert_allclose(pelvic, x['errors'][:, 1], atol=1e-12, rtol=0)
            flags = x['errors'][:, 1:]>np.array([thresholds['pelvis_z_error_m'],
                thresholds['torso_projected_gravity_l2'], thresholds['end_effector_z_error_m']])
            np.testing.assert_array_equal(flags, x['termination_flags'])
            complete = count == row['reference_frames']
            passed = complete and not flags.any() and float(body_error.mean()) < thresholds['full_clip_mean_mpjpe_m']
            if bool(passed) != row['auxiliary_pass'] or bool(complete) != row['full_clip_executed']:
                raise ValueError('Receipt did not match the complete actual episode')
            verified.append({'index': index, 'seed': seed, 'raw_sha256': row['raw_sha256'],
                             'control_frames': count, 'pass': bool(passed)})
    accepted = [i for i in range(len(entries)) if all(by_key[i, s]['auxiliary_pass'] for s in seeds)]
    if accepted != summary['all_seed_pass_indices']:
        raise ValueError('All-seed selection mismatch')
    selected = copy.deepcopy(manifest)
    selected['purpose'] = 'pending_phygile_gmt_validation'
    selected['splits'] = {s: [] for s in manifest['splits']}
    for i in accepted:
        split, entry = entries[i]
        row = copy.deepcopy(entry)
        for key in ['features', 'native', 'codec_metadata']:
            row[key] = str((a.manifest.parent/entry[key]).resolve(strict=True))
        if sha(row['features']) != entry['sha256'] or sha(row['native']) != entry['native_sha256']:
            raise ValueError('An original paired asset changed')
        row['auxiliary_physical_filter'] = {'tracker': 'official frozen external ScaleBFM M',
            'checkpoint_sha256': summary['checkpoint_sha256'], 'all_seeds': seeds,
            'all_seeds_pass': True, 'paper_phygile_gmt_filter': False,
            'record_sha256': [by_key[i, s]['raw_sha256'] for s in seeds]}
        selected['splits'][split].append(row)
    selected['auxiliary_filter'] = {'summary_sha256': sha(a.records/'summary.json'),
        'thresholds': thresholds, 'tracker_role': summary['filter_role'],
        'all_seeds_required': True, 'author_filter_verified': False,
        'training_ready': False, 'source_data_deleted': False}
    arrays = [np.load(e['features'], allow_pickle=False) for e in selected['splits']['train']]
    if arrays:
        norm = BlockNormalizer.fit(arrays)
        np.savez(a.output/'normalization_auxiliary_train_only.npz', mean=norm.mean, std=norm.std)
    (a.output/'manifest_auxiliary_passed.json').write_text(json.dumps(selected, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    result = {'status': 'ALL_EPISODES_VERIFIED_AUXILIARY_SUBSET_WRITTEN', 'verified_episodes': len(verified),
        'source_sequences': len(entries), 'all_seed_pass_sequences': len(accepted),
        'selected_indices': accepted, 'split_sequences': {s: len(v) for s, v in selected['splits'].items()},
        'split_frames': {s: sum(e['frames'] for e in v) for s, v in selected['splits'].items()},
        'unique_passed_sources': len({e['source_group'] for v in selected['splits'].values() for e in v}),
        'data_arrays_copied': False, 'original_data_deleted': False,
        'trained_generator_started': False, 'paper_training_ready': False,
        'blocking_facts': ['External tracker does not replace the paper Stage-II GMT filter.',
            'Only the existing small SSM subset is covered; remaining HumanML sources/model dependencies unresolved.',
            'No complete source-disjoint validation set, paper training corpus or author evaluation assets.'],
        'inputs': {'manifest_sha256': sha(a.manifest), 'screen_summary_sha256': sha(a.records/'summary.json')},
        'output_manifest_sha256': sha(a.output/'manifest_auxiliary_passed.json'), 'verified_records': verified,
        'script_sha256': sha(__file__)}
    (a.output/'verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'verified_records'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['manifest', 'records', 'phygile-code', 'output']:
        p.add_argument('--'+name, required=True, type=Path)
    run(p.parse_args())

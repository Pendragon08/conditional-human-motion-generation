"""Build isolated constant-height floor-aligned trajectory variants.

Assumes each clip contains a moment of foot contact with z=0 and uses the
previously measured MuJoCo foot-sphere geometry. This is an engineering
counterfactual, not a ground-truth calibration or the authors' method.
Nothing is accepted for training and all original inputs are preserved.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from feature263 import encode
from g1_fk import URDFFK
from training_stats import training_stats

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT = ROOT / 'eight_projected_preflight_20261031'
MEASURE = ROOT / 'eight_static_robot_quality_20261031'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    diagnostic = json.loads((MEASURE / 'report.json').read_text())
    audit = json.loads((MEASURE / 'audit_report.json').read_text())
    parent_config = json.loads((INPUT / 'preflight_config.json').read_text())
    if (sha(MEASURE / 'report.json') != audit['report_sha256'] or
        sha(INPUT / 'manifest.json') != diagnostic['manifest_sha256'] or
        sha(INPUT / 'state_manifest.json') != diagnostic['state_manifest_sha256'] or
        sha(URDF) != parent_config['state_urdf_sha256'] or
        not np.allclose(diagnostic['floor_position'], [0., 0., 0.], atol=1e-10, rtol=0) or
        not np.allclose(diagnostic['floor_normal'], [0., 0., 1.], atol=1e-10, rtol=0)):
        raise ValueError('source diagnostic or assumed plane changed')
    records = json.loads((INPUT / 'manifest.json').read_text())['records']
    original_states = {x['sample_id']: x for x in json.loads((INPUT / 'state_manifest.json').read_text())['records']}
    cases = {x['sample_id']: x for x in diagnostic['cases']}
    if len(records) != 8 or set(cases) != set(original_states) or set(cases) != {r['sample_id'] for r in records}:
        raise ValueError('case coverage mismatch')
    fk = URDFFK(URDF)
    new_states, outputs, train_arrays, spatial = [], [], [], []
    for rec in records:
        ident = rec['sample_id']; state = original_states[ident]; measured = cases[ident]
        state_path, feature_path = INPUT / state['state_path'], INPUT / rec['feature_path']
        if (sha(state_path) != state['state_sha256'] or sha(feature_path) != rec['feature_sha256'] or
            measured['state_sha256'] != state['state_sha256'] or measured['feature_sha256'] != rec['feature_sha256']):
            raise ValueError('state provenance changed')
        with np.load(state_path, allow_pickle=False) as f:
            root, joints = f['root_wxyz_xyz'].copy(), f['joint_pos'].copy()
            times, names = f['sample_times'].copy(), f['joint_names'].copy()
        with np.load(feature_path, allow_pickle=False) as f:
            features_old = f['features'].copy()
        if root.shape != (len(joints), 7) or joints.shape[1:] != (27,) or len(times) != len(joints) or len(features_old) + 1 != len(joints):
            raise ValueError('bad source dimensions')
        lowest = min(measured['foot_gap_min_m'])
        if not np.isfinite(lowest):
            raise ValueError('invalid foot gap')
        shift = -float(lowest)
        root_new = root.copy()
        root_new[:, 6] += shift
        old_points = fk.forward(root[:, 4:], root[:, [1, 2, 3, 0]], joints, list(names))
        new_points = fk.forward(root_new[:, 4:], root_new[:, [1, 2, 3, 0]], joints, list(names))
        change = new_points - old_points
        translation_error = float(np.max(np.abs(change - np.array([0., 0., shift]))))
        if translation_error > 1e-6:
            raise ValueError('FK is not invariant to constant translation')
        features = encode(new_points, foot_velocity_squared_threshold=state['foot_velocity_squared_threshold'])
        feature_delta = float(np.max(np.abs(features.astype(np.float64) - features_old.astype(np.float64))))
        flags_changed = int(np.count_nonzero(features[:, -4:] != features_old[:, -4:]))
        new_state = HERE / state['state_path']; new_feature = HERE / rec['feature_path']
        with new_state.open('xb') as f:
            np.savez_compressed(f, root_wxyz_xyz=root_new, joint_pos=joints, sample_times=times, joint_names=names)
        with new_feature.open('xb') as f:
            np.savez_compressed(f, features=features)
        previous_feature_sha, previous_state_sha = rec['feature_sha256'], state['state_sha256']
        rec = dict(rec, feature_sha256=sha(new_feature), parent_feature_sha256=previous_feature_sha,
                   training_ready=False, caption_accepted=False)
        outputs.append(rec)
        new_states.append({'sample_id': ident, 'feature_sha256': rec['feature_sha256'],
                           'state_path': state['state_path'], 'state_sha256': sha(new_state),
                           'parent_state_sha256': previous_state_sha,
                           'foot_velocity_squared_threshold': state['foot_velocity_squared_threshold']})
        if rec['split'] == 'train':
            train_arrays.append(('train', features))
            with torch.no_grad():
                spatial.append(recover_from_ric(torch.from_numpy(features), 22).numpy().reshape(-1, 66).astype(np.float64))
        measured['variant'] = {'root_shift_z_m': shift, 'parent_foot_gaps_min_m': measured['foot_gap_min_m'],
                               'fk_translation_max_error_m': translation_error,
                               'feature_max_abs_change': feature_delta, 'foot_proxy_values_changed': flags_changed,
                               'feature_sha256': rec['feature_sha256'], 'state_sha256': new_states[-1]['state_sha256']}
    stats = training_stats(train_arrays, min_std=.001); spatial = np.concatenate(spatial)
    with (HERE / 'stats.npz').open('xb') as f:
        np.savez_compressed(f, feature_mean=stats['mean'], feature_std=stats['std'],
                            spatial_mean=spatial.mean(0), spatial_std=np.maximum(spatial.std(0), .001))
    for name, payload in [('manifest.json', {'records': outputs, 'training_ready': False}),
                          ('state_manifest.json', {'records': new_states, 'training_ready': False})]:
        with (HERE / name).open('x', encoding='utf8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')
    config = json.loads((INPUT / 'preflight_config.json').read_text())
    config.update({'dataset_root': str(HERE), 'manifest': str(HERE / 'manifest.json'),
                   'stats': str(HERE / 'stats.npz'), 'state_root': str(HERE),
                   'state_manifest': str(HERE / 'state_manifest.json'), 'run_dir': str(HERE), 'device': 'cpu',
                   'purpose': 'unaccepted hypothetical z=0 floor-aligned counterfactual'})
    with (HERE / 'preflight_config.json').open('x', encoding='utf8') as f:
        json.dump(config, f, indent=2, allow_nan=False); f.write('\n')
    out = {'parent_manifest_sha256': diagnostic['manifest_sha256'],
           'parent_state_manifest_sha256': diagnostic['state_manifest_sha256'],
           'floor_diagnostic_sha256': sha(MEASURE / 'report.json'), 'urdf_sha256': sha(URDF),
           'script_sha256': sha(Path(__file__)), 'manifest_sha256': sha(HERE / 'manifest.json'),
           'module_sha256': {x: sha(HERE / x) for x in ('feature263.py', 'g1_fk.py', 'training_stats.py')},
           'state_manifest_sha256': sha(HERE / 'state_manifest.json'), 'stats_sha256': sha(HERE / 'stats.npz'),
           'train_frames': stats['frame_count'], 'cases': [cases[r['sample_id']]['variant'] | {'sample_id': r['sample_id']} for r in records],
           'training_ready': False,
           'limitations': ['Assumes each motion touches z=0, not independently calibrated ground or observed contact.',
                           'No modification of velocities or joint rotations; no dynamics or collision resolution.',
                           'Retargeting, source identity, captions, real FPS, physics and training unaccepted.']}
    with (HERE / 'report.json').open('x', encoding='utf8') as f:
        json.dump(out, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'cases': len(outputs), 'train_frames': stats['frame_count'], 'training_ready': False}))


if __name__ == '__main__':
    main()

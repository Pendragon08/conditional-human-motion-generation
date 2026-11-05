"""Create a separate, bounded joint-limit projection preview; never train.

This engineering repair is not the unpublished v2 method and does not confer
physical, caption, source, timing, or training acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from feature263 import encode
from g1_fk import URDFFK
from g1_joint_limit_gate import check_joint_limits, read_joint_limits
from training_stats import training_stats
from data_loaders.humanml.scripts.motion_process import recover_from_ric


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def project(q, names, limits, max_correction):
    if isinstance(max_correction, bool) or not np.isfinite(max_correction) or max_correction <= 0:
        raise ValueError('maximum correction must be finite and positive')
    q = np.asarray(q, dtype=np.float64)
    before = check_joint_limits(q, names, limits, tolerance_rad=0)
    if before['max_excess_rad'] > max_correction:
        raise ValueError('required correction exceeds configured bound')
    lo = np.asarray([limits[n][0] for n in names]); hi = np.asarray([limits[n][1] for n in names])
    fixed = np.minimum(np.maximum(q, lo), hi)
    after = check_joint_limits(fixed, names, limits, tolerance_rad=0)
    if not after['joint_limits_passed']:
        raise RuntimeError('projection did not satisfy URDF limits')
    return fixed, before, after


def source_file(root, relative, expected):
    relative = Path(relative)
    if relative.is_absolute(): raise ValueError('absolute input path')
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected:
        raise ValueError('input path or hash mismatch')
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--feature-root', type=Path, required=True)
    ap.add_argument('--state-manifest', type=Path, required=True)
    ap.add_argument('--state-root', type=Path, required=True)
    ap.add_argument('--urdf', type=Path, required=True)
    ap.add_argument('--urdf-sha256', required=True)
    ap.add_argument('--max-correction', type=float, required=True)
    ap.add_argument('--foot-threshold', type=float, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    feature_root = args.feature_root.resolve(strict=True); state_root = args.state_root.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True); evidence_path = args.state_manifest.resolve(strict=True)
    urdf = args.urdf.resolve(strict=True)
    manifest_sha, evidence_sha, urdf_sha = sha(manifest_path), sha(evidence_path), sha(urdf)
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    evidence = json.loads(evidence_path.read_text(encoding='utf8'))
    if evidence['preview_manifest_sha256'] != manifest_sha or urdf_sha != args.urdf_sha256:
        raise ValueError('manifest or URDF mismatch')
    states = {r['sample_id']: r for r in evidence['records']}
    records = manifest['records']
    if len(states) != len(records) or set(states) != {r['sample_id'] for r in records}:
        raise ValueError('state evidence must exactly cover the feature manifest')
    limits = read_joint_limits(urdf); robot = URDFFK(urdf)
    args.output.mkdir(parents=False, exist_ok=False)
    updated, state_rows, metrics, arrays, decoded_train = [], [], [], [], []
    for rec in records:
        name = rec['sample_id']
        if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('unsafe sample ID')
        state_row = states[name]
        if state_row['feature_sha256'] != rec['feature_sha256']:
            raise ValueError('state and feature identity differ')
        feature_file = source_file(feature_root, rec['feature_path'], rec['feature_sha256'])
        state_file = source_file(state_root, state_row['state_path'], state_row['state_sha256'])
        with np.load(feature_file, allow_pickle=False) as data:
            original_features = data['features'].copy()
        with np.load(state_file, allow_pickle=False) as data:
            root = data['root_wxyz_xyz'].copy(); q = data['joint_pos'].copy()
            times = data['sample_times'].copy(); names_array = data['joint_names'].copy()
        names = [str(x) for x in names_array]
        if (root.shape != (len(q), 7) or times.shape != (len(q),) or len(q) != len(original_features) + 1 or
            not np.isfinite(root).all() or not np.isfinite(times).all() or
            not np.allclose(np.diff(times), .05, atol=1e-8, rtol=0)):
            raise ValueError('invalid state/feature timing')
        repaired, before, after = project(q, names, limits, args.max_correction)
        original_points = robot.forward(root[:, 4:], root[:, [1, 2, 3, 0]], q, names)
        original_encoded = encode(original_points, foot_velocity_squared_threshold=args.foot_threshold)
        if original_encoded.shape != original_features.shape or not np.allclose(original_encoded, original_features, atol=1e-5, rtol=0):
            raise ValueError('original features do not match state/encoder')
        repaired_points = robot.forward(root[:, 4:], root[:, [1, 2, 3, 0]], repaired, names)
        features = encode(repaired_points, foot_velocity_squared_threshold=args.foot_threshold)
        delta = repaired - q
        point_delta = np.linalg.norm(repaired_points - original_points, axis=-1)
        feature_delta = np.abs(features.astype(np.float64) - original_features.astype(np.float64))
        feature_path = args.output / (name + '.npz'); state_path = args.output / (name + '_state.npz')
        with feature_path.open('xb') as out:
            np.savez_compressed(out, features=features, sample_times=times)
        with state_path.open('xb') as out:
            np.savez_compressed(out, root_wxyz_xyz=root, joint_pos=repaired, sample_times=times, joint_names=names_array)
        row = {**rec, 'feature_path': feature_path.name, 'feature_sha256': sha(feature_path),
               'parent_feature_sha256': rec['feature_sha256'], 'caption_accepted': False, 'training_ready': False}
        updated.append(row)
        state_rows.append({'sample_id': name, 'feature_sha256': row['feature_sha256'],
                           'state_path': state_path.name, 'state_sha256': sha(state_path),
                           'foot_velocity_squared_threshold': args.foot_threshold,
                           'parent_state_sha256': state_row['state_sha256']})
        metrics.append({'sample_id': name, 'split': rec['split'], 'before': before, 'after': after,
                        'joint_values_changed': int(np.count_nonzero(delta)),
                        'max_joint_correction_rad': float(np.max(np.abs(delta))),
                        'max_point_displacement_m': float(point_delta.max()),
                        'max_angular_velocity_change_rad_s': float(np.max(np.abs(np.diff(delta, axis=0))) / .05),
                        'max_angular_acceleration_change_rad_s2': float(np.max(np.abs(np.diff(delta, n=2, axis=0))) / .05**2),
                        'max_feature_difference': float(feature_delta.max()),
                        'foot_proxy_values_changed': int(np.count_nonzero(features[:, -4:] != original_features[:, -4:]))})
        arrays.append((rec['split'], features))
        if rec['split'] == 'train':
            with torch.no_grad():
                decoded_train.append(recover_from_ric(torch.from_numpy(features), 22).numpy().reshape(-1, 66).astype(np.float64))
        if sha(feature_file) != rec['feature_sha256'] or sha(state_file) != state_row['state_sha256']:
            raise RuntimeError('input changed during processing')
    stats = training_stats(arrays, min_std=.001); spatial = np.concatenate(decoded_train)
    with (args.output / 'stats.npz').open('xb') as f:
        np.savez_compressed(f, feature_mean=stats['mean'], feature_std=stats['std'],
                            spatial_mean=spatial.mean(0), spatial_std=np.maximum(spatial.std(0), .001))
    new_manifest = {'records': updated, 'training_ready': False, 'purpose': 'bounded projection preview only',
                    'parent_manifest_sha256': manifest_sha}
    new_state_manifest = {'records': state_rows, 'training_ready': False, 'parent_state_manifest_sha256': evidence_sha}
    for filename, value in [('manifest.json', new_manifest), ('state_manifest.json', new_state_manifest)]:
        with (args.output / filename).open('x', encoding='utf8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    if (sha(manifest_path), sha(evidence_path), sha(urdf)) != (manifest_sha, evidence_sha, urdf_sha):
        raise RuntimeError('source manifest or URDF changed')
    result = {'source_manifest_sha256': manifest_sha, 'source_state_manifest_sha256': evidence_sha,
              'urdf_sha256': urdf_sha, 'max_correction_rad': args.max_correction,
              'foot_threshold': args.foot_threshold, 'cases': metrics,
              'manifest_sha256': sha(args.output / 'manifest.json'),
              'state_manifest_sha256': sha(args.output / 'state_manifest.json'),
              'stats_sha256': sha(args.output / 'stats.npz'), 'train_frames': stats['frame_count'],
              'training_ready': False,
              'limitations': ['Bounded per-sample projection is an engineering approximation, not the v2 method.',
                              'Foot velocity is not physical contact; no collision or dynamics verified.',
                              'Caption semantics, source identity and actual sample time are not accepted.']}
    with (args.output / 'report.json').open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'cases': len(metrics), 'changed_cases': sum(x['joint_values_changed'] > 0 for x in metrics),
                      'largest_correction_rad': max(x['max_joint_correction_rad'] for x in metrics),
                      'largest_point_displacement_m': max(x['max_point_displacement_m'] for x in metrics),
                      'training_ready': False}))


if __name__ == '__main__': main()

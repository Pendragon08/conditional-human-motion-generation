"""Recompute FK linkage and joint-limit diagnostics for the seven-record preview.

This is an isolated audit: matching feature points and numerical joint bounds
are necessary checks, not complete source, caption or physical acceptance.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from feature263 import canonicalize_z_up
from g1_fk import URDFFK
from g1_joint_limit_gate import check_joint_limits, read_joint_limits

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PREVIEW = ROOT / 'seven_candidate_technical_preview_20261031'
EARLIER = ROOT / 'six_joint_diagnostics_20261031'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    torch.set_num_threads(1)
    manifest_path = PREVIEW / 'experimental_manifest_not_for_training.json'
    audit = json.loads((PREVIEW / 'independent_audit.json').read_text())
    old = json.loads((EARLIER / 'report.json').read_text())
    old_check = json.loads((EARLIER / 'independent_audit.json').read_text())
    if sha(manifest_path) != audit['manifest_sha256'] or sha(EARLIER / 'report.json') != old_check['input_report_sha256']:
        raise ValueError('preview or trajectory audit changed')
    if sha(URDF) != old['urdf_sha256']:
        raise ValueError('URDF changed')
    old_by_id = {row['sample_id']: row for row in old['cases']}
    fk, limits, rows = URDFFK(URDF), read_joint_limits(URDF), []
    for rec in json.loads(manifest_path.read_text())['records']:
        mid, path = rec['sample_id'], PREVIEW / rec['feature_path']
        if sha(path) != rec['feature_sha256']:
            raise ValueError('feature changed')
        with np.load(path, allow_pickle=False) as data:
            features = data['features'].copy()
            if mid == 'ACCAD_004347':
                state = data['filtered_20hz'].copy()
                root, q = state[:, :7], state[:, 7:]
                names = [str(n) for n in data['joint_names']]
            elif mid.startswith('BML_'):
                root, q = data['root_wxyz_xyz'].copy(), data['joint_pos'].copy()
                names = [str(n) for n in data['joint_names']]
                grid = data['source_times'].copy()
            else:
                grid = data['time_grid'].copy()
        if mid != 'ACCAD_004347':
            folder = (ROOT / 'pyroki_ssm_000008_global_20261031' if mid == '000008' else
                      ROOT / 'bml_joint_smoothness_20261031' / mid[4:] if mid.startswith('BML_') else
                      ROOT / 'ssm_three_global_20261031' / mid)
            trajectory = folder / 'trajectory.npz'
            if sha(trajectory) != old_by_id[mid]['trajectory_sha256']:
                raise ValueError('trajectory changed')
            with np.load(trajectory, allow_pickle=False) as source:
                times = source['source_times'].copy()
                source_q = source['joint_pos'].copy()
                source_root = source['root_wxyz_xyz'].copy()
                source_names = [str(n) for n in source['joint_names']]
            if grid[0] < times[0] - 1e-10 or grid[-1] > times[-1] + 1e-10:
                raise ValueError('invalid feature time grid')
            interpolated = np.column_stack([np.interp(grid, times, source_q[:, j]) for j in range(27)])
            if mid.startswith('BML_'):
                if names != source_names or not np.allclose(q, interpolated, atol=1e-10, rtol=0):
                    raise ValueError('BML joint trajectory mismatch')
            else:
                names, q = source_names, interpolated
                root = np.empty((len(grid), 7))
                root[:, 4:] = np.column_stack([np.interp(grid, times, source_root[:, j]) for j in (4, 5, 6)])
                rotation = Slerp(times, Rotation.from_quat(source_root[:, [1, 2, 3, 0]]))(grid).as_quat()
                root[:, :4] = rotation[:, [3, 0, 1, 2]]
        if len(q) != len(features) + 1 or root.shape != (len(q), 7):
            raise ValueError('robot and feature lengths differ')
        points = fk.forward(root[:, 4:], root[:, [1, 2, 3, 0]], q, names)
        with torch.no_grad():
            decoded = recover_from_ric(torch.from_numpy(features), 22).numpy()
        error = float(np.linalg.norm(decoded - canonicalize_z_up(points)[:-1], axis=-1).max())
        if not np.isfinite(error) or error > 1e-4:
            raise ValueError('robot state does not match decoded features: {}'.format(mid))
        joint_check = check_joint_limits(q, names, limits, tolerance_rad=1e-6)
        rows.append({'sample_id': mid, 'split': rec['split'], 'feature_sha256': rec['feature_sha256'],
                     'fk_decode_max_error_m': error, 'joint_check': joint_check, 'training_ready': False})
    result = {'manifest_sha256': sha(manifest_path), 'urdf_sha256': sha(URDF), 'cases': rows,
              'joint_check_failed_samples': [r['sample_id'] for r in rows if not r['joint_check']['joint_limits_passed']],
              'training_ready': False,
              'limitations': ['Uses the project FK and decoder; not an independent complete re-encoding.',
                              '1e-6 rad is a numeric allowance, not a paper feasibility threshold.',
                              'Does not validate true time, source identity, caption semantics or physical quality.',
                              'This audit is not yet integrated with the guarded training entry.']}
    with (HERE / 'report.json').open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'cases': len(rows), 'failed': result['joint_check_failed_samples'], 'training_ready': False}))


if __name__ == '__main__':
    main()

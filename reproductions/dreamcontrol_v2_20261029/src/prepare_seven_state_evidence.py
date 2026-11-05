"""Materialize non-training sampled-state evidence for the seven-record preview."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PREVIEW = ROOT / 'seven_candidate_technical_preview_20261031'
OLD = ROOT / 'six_joint_diagnostics_20261031'
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    preview_path = PREVIEW / 'experimental_manifest_not_for_training.json'
    preview_audit = json.loads((PREVIEW / 'independent_audit.json').read_text())
    old_path = OLD / 'report.json'
    old_audit = json.loads((OLD / 'independent_audit.json').read_text())
    if sha(preview_path) != preview_audit['manifest_sha256'] or sha(old_path) != old_audit['input_report_sha256']:
        raise ValueError('input audit mismatch')
    old = {row['sample_id']: row for row in json.loads(old_path.read_text())['cases']}
    records = []
    for rec in json.loads(preview_path.read_text())['records']:
        sid = rec['sample_id']
        feature = PREVIEW / rec['feature_path']
        if sha(feature) != rec['feature_sha256']:
            raise ValueError('feature hash mismatch')
        with np.load(feature, allow_pickle=False) as data:
            if sid == 'ACCAD_004347':
                state = data['filtered_20hz'].copy()
                root, q, times = state[:, :7], state[:, 7:], data['frames_20hz'].astype(float) / 120.
                names = data['joint_names'].copy()
            elif sid.startswith('BML_'):
                root, q, times = data['root_wxyz_xyz'].copy(), data['joint_pos'].copy(), data['source_times'].copy()
                names = data['joint_names'].copy()
            else:
                times = data['time_grid'].copy()
        if sid != 'ACCAD_004347':
            directory = (ROOT / 'pyroki_ssm_000008_global_20261031' if sid == '000008' else
                         ROOT / 'bml_joint_smoothness_20261031' / sid[4:] if sid.startswith('BML_') else
                         ROOT / 'ssm_three_global_20261031' / sid)
            trajectory = directory / 'trajectory.npz'
            if sha(trajectory) != old[sid]['trajectory_sha256']:
                raise ValueError('trajectory hash mismatch')
            with np.load(trajectory, allow_pickle=False) as data:
                t = data['source_times'].copy()
                src_root, src_q, src_names = data['root_wxyz_xyz'].copy(), data['joint_pos'].copy(), data['joint_names'].copy()
            if sid.startswith('BML_'):
                if not np.array_equal(names, src_names) or not np.allclose(q, src_q, atol=1e-10, rtol=0):
                    raise ValueError('BML state mismatch')
            else:
                q = np.column_stack([np.interp(times, t, src_q[:, j]) for j in range(27)])
                root = np.empty((len(times), 7))
                root[:, 4:] = np.column_stack([np.interp(times, t, src_root[:, j]) for j in (4, 5, 6)])
                quat = Slerp(t, Rotation.from_quat(src_root[:, [1, 2, 3, 0]]))(times).as_quat()
                root[:, :4] = quat[:, [3, 0, 1, 2]]
                names = src_names
        target = HERE / (sid + '_state.npz')
        with target.open('xb') as output:
            np.savez_compressed(output, root_wxyz_xyz=root, joint_pos=q, sample_times=times, joint_names=names)
        records.append({'sample_id': sid, 'feature_sha256': rec['feature_sha256'],
                        'state_path': target.name, 'state_sha256': sha(target)})
    with (HERE / 'state_manifest.json').open('x') as output:
        json.dump({'records': records, 'preview_manifest_sha256': sha(preview_path), 'training_ready': False}, output, indent=2)
        output.write('\n')
    print(json.dumps({'records': len(records), 'training_ready': False}))


if __name__ == '__main__':
    main()

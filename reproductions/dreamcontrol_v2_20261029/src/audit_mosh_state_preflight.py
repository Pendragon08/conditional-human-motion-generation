"""Run the existing full state/feature preflight on the cropped MoSh candidate."""
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_robot_states import audit_robot_states

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
CROP = ROOT / 'mosh_000798_crop_candidate_20261031'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    r = json.loads((CROP / 'report.json').read_text())
    audit = json.loads((CROP / 'audit_report.json').read_text())
    feature = CROP / 'candidate.npz'
    if sha(feature) != r['output_sha256'] or sha(feature) != audit['output_sha256']:
        raise ValueError('source changed')
    with np.load(feature, allow_pickle=False) as f:
        state = f['filtered_20hz'].copy()
        frames = f['frames_20hz'].copy()
        names = f['joint_names'].copy()
    with (HERE / 'state.npz').open('xb') as f:
        np.savez_compressed(f, root_wxyz_xyz=state[:, :7], joint_pos=state[:, 7:],
                            sample_times=frames.astype(np.float64) / 120., joint_names=names)
    records = [{'sample_id': 'MOSH_000798_crop', 'feature_path': 'candidate.npz', 'feature_sha256': sha(feature),
                'state_path': 'state.npz', 'state_sha256': sha(HERE / 'state.npz'),
                'foot_velocity_squared_threshold': .002}]
    result = audit_robot_states(records, feature_root=CROP, state_root=HERE, urdf_path=URDF,
                                tolerance_rad=1e-6, fk_decode_tolerance_m=1e-4, reencoding_tolerance=1e-5)
    result['candidate_report_sha256'] = sha(CROP / 'report.json')
    result['nominal_start_seconds'] = float(frames[0] / 120.)
    result['nominal_end_seconds'] = float(frames[-1] / 120.)
    with (HERE / 'report.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

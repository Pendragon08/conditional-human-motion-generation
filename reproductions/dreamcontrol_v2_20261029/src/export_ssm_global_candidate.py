"""Create one isolated SSM robot feature/text candidate, explicitly not training-ready.

Consumes the audited full-trajectory solve. Keeps exact source frame/time and
caption provenance, with no official-split modification or acceptance claim.
"""
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from feature263 import canonicalize_z_up, encode
from g1_fk import URDFFK

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOLVE = ROOT / 'pyroki_ssm_000008_global_20261031'
HUMANML = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    report = json.loads((SOLVE / 'report.json').read_text())
    audit = json.loads((SOLVE / 'audit_report.json').read_text())
    trajectory = SOLVE / 'trajectory.npz'
    if (sha(SOLVE / 'report.json') != audit['report_sha256'] or
            sha(trajectory) != report['artifact_sha256'] or
            sha(trajectory) != audit['artifact_sha256'] or
            sha(URDF) != report['urdf_sha256'] or audit['frames_verified'] != 42):
        raise ValueError('audited trajectory provenance mismatch')
    with np.load(trajectory, allow_pickle=False) as data:
        t = np.asarray(data['source_times'], dtype=np.float64)
        source_frames = data['source_frames'].copy()
        source_indices = data['sample_indices'].copy()
        root = data['root_wxyz_xyz'].astype(np.float64)
        q = data['joint_pos'].astype(np.float64)
        names = [str(x) for x in data['joint_names']]
    if t.shape != (42,) or np.any(np.diff(t) <= 0) or q.shape != (42, 27):
        raise ValueError('invalid source timeline')
    # Strict 20 Hz anchored at the first available solution, no extrapolation.
    count = int(np.floor((t[-1] - t[0]) * 20.0)) + 1
    grid = t[0] + np.arange(count) / 20.0
    if count < 3 or grid[-1] > t[-1]: raise ValueError('invalid output grid')
    positions = np.column_stack([np.interp(grid, t, root[:, i]) for i in (4, 5, 6)])
    joints = np.column_stack([np.interp(grid, t, q[:, i]) for i in range(27)])
    quat = Slerp(t, Rotation.from_quat(root[:, [1, 2, 3, 0]]))(grid).as_quat()
    points = URDFFK(URDF).forward(positions, quat, joints, names)
    features = encode(points, foot_velocity_squared_threshold=.002)
    with torch.no_grad(): decoded = recover_from_ric(torch.from_numpy(features), 22).numpy()
    error = float(np.linalg.norm(decoded - canonicalize_z_up(points)[:-1], axis=-1).max())
    if error > 1e-4: raise ValueError('feature decode mismatch')
    archive = HUMANML / 'texts.zip'
    archive_hash = sha(archive)
    with zipfile.ZipFile(archive) as z:
        member = z.read('texts/000008.txt')
    captions = []
    for i, line in enumerate(member.decode('utf8').splitlines(), 1):
        fields = line.split('#')
        if len(fields) != 4 or not fields[0] or fields[2:] != ['0.0', '0.0']:
            raise ValueError('unexpected caption or timing tags')
        captions.append({'line': i, 'caption': fields[0], 'tag_strings': fields[2:]})
    if len(captions) != 4: raise ValueError('unexpected caption count')
    splits = {name: set((HUMANML / (name + '.txt')).read_text().splitlines())
              for name in ('train', 'val', 'test')}
    matches = [name for name, ids in splits.items() if '000008' in ids]
    if matches != ['train'] or sha(archive) != archive_hash:
        raise ValueError('split or caption archive changed')
    out = HERE / 'candidate.npz'
    with out.open('xb') as f:
        np.savez_compressed(f, features=features, time_grid=grid, decoded_points=decoded,
            points_z_up_world=points, root_position=positions, root_quaternion_xyzw=quat,
            joint_pos=joints, joint_names=np.asarray(names), original_sample_indices=source_indices,
            original_source_frames=source_frames, original_times=t)
    result = {'motion_id': '000008', 'source_path': './pose_data/SSM_synced/20161014_50033/ATUSquat_sync_poses.npy',
        'training_ready': False, 'official_split_for_id': 'train',
        'source_trajectory_sha256': sha(trajectory), 'solve_report_sha256': sha(SOLVE / 'report.json'),
        'solve_audit_sha256': sha(SOLVE / 'audit_report.json'), 'urdf_sha256': sha(URDF),
        'caption_zip_sha256': archive_hash, 'caption_member': 'texts/000008.txt',
        'caption_member_sha256': hashlib.sha256(member).hexdigest(), 'captions': captions,
        'official_split_file_sha256': sha(HUMANML / 'train.txt'),
        'sample_rate_hz': 20, 'input_points': len(t), 'output_points': len(grid),
        'feature_rows': len(features), 'source_first_time_s': float(t[0]),
        'source_last_time_s': float(t[-1]), 'output_first_time_s': float(grid[0]),
        'output_last_time_s': float(grid[-1]), 'tail_not_sampled_s': float(t[-1] - grid[-1]),
        'decode_max_error_m': error, 'foot_threshold_squared': .002,
        'output_sha256': sha(out), 'script_sha256': sha(__file__),
        'limitations': ['one short candidate from one recording; no validation or test split',
            'caption tags are whole-sequence convention, but semantic alignment after cropping is unverified',
            '20 Hz linear/Slerp resampling and 263D threshold are experimental',
            'source model equivalence, retarget quality, contact, collision, dynamics and license unaccepted',
            'no training dataset, training statistics, or paper-equivalent method']}
    with (HERE / 'report.json').open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()

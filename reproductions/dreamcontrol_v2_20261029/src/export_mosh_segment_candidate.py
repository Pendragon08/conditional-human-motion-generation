"""Create an explicitly cropped, unaccepted MoSh robot feature candidate.

The original windows, including unstable windows, remain untouched. This
diagnostic does not establish that the crop preserves caption semantics or
that its robot trajectory is physically plausible.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from blend_robot_windows import blend_windows
from resample_g1_27 import resample_27
from feature263 import canonicalize_z_up, encode
from g1_fk import URDFFK

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
RUN = ROOT / 'mosh_000798_gpu_resume_20261031'
HUMAN = ROOT / 'mosh_000798_fullrate_20261031/output'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    result = json.loads((RUN / 'report.json').read_text())
    check = json.loads((RUN / 'audit_report.json').read_text())
    human = json.loads((HUMAN / 'human_report.json').read_text())
    if (sha(RUN / 'report.json') != check['report_sha256'] or
        sha(HUMAN / 'human_120hz.npz') != check['source_sha256'] or
        sha(URDF) != check['urdf_sha256'] or check['windows_verified'] != 253 or
        check['covered_indices'] != [1, 1014] or
        sha(RUN / 'windows.jsonl') != result['rows_sha256']):
        raise ValueError('provenance mismatch')
    rows = [json.loads(x) for x in (RUN / 'windows.jsonl').read_text().splitlines()]
    windows = []
    # First 31 windows include four observed 2.68–3.26 rad disagreements.
    # Start after them, and do not import window 30's values into the crop.
    for row in rows[31:]:
        path = RUN / row['artifact']
        if sha(path) != row['artifact_sha256']:
            raise ValueError('window changed')
        with np.load(path, allow_pickle=False) as f:
            windows.append({'frames': f['source_frames'].copy(), 'root_wxyz_xyz': f['root_wxyz_xyz'].copy(),
                            'joint_pos': f['joint_pos'].copy(), 'joint_names': f['joint_names'].copy()})
    merged = blend_windows(windows)
    frames = merged['frames']
    np.testing.assert_array_equal(frames, np.arange(125, 1015))
    start = int(np.flatnonzero(frames == 126)[0])
    filtered = resample_27(merged['blended'][start:], source_fps=120., target_fps=20.)
    grid = frames[start::6]
    if filtered.shape != (149, 34) or grid.tolist() != list(range(126, 1015, 6)):
        raise ValueError('unexpected grid')
    names = list(merged['joint_names'])
    points = URDFFK(URDF).forward(filtered[:, 4:7], filtered[:, [1, 2, 3, 0]], filtered[:, 7:], names)
    features = encode(points, foot_velocity_squared_threshold=.002)
    with torch.no_grad():
        decoded = recover_from_ric(torch.from_numpy(features), 22).numpy()
    err = float(np.linalg.norm(decoded - canonicalize_z_up(points)[:-1], axis=-1).max())
    if features.shape != (148, 263) or not np.isfinite(features).all() or err > 1e-4:
        raise ValueError('feature self-check failed')
    limits = {}
    for joint in ET.parse(URDF).getroot().findall('joint'):
        if joint.get('type') == 'continuous':
            limits[joint.get('name')] = (-np.inf, np.inf)
        elif joint.get('type') == 'revolute':
            limit = joint.find('limit')
            limits[joint.get('name')] = (float(limit.get('lower')), float(limit.get('upper')))
    if set(names) != set(limits):
        raise ValueError('joint names differ')
    low = np.array([limits[x][0] for x in names])
    high = np.array([limits[x][1] for x in names])
    violation = lambda x: float(np.maximum(np.maximum(low - x, x - high), 0).max())
    out = HERE / 'candidate.npz'
    with out.open('xb') as f:
        np.savez_compressed(f, features=features, points_z_up_world=points, decoded_points=decoded,
                            raw_frames=frames, blended_120hz=merged['blended'], frames_20hz=grid,
                            filtered_20hz=filtered, joint_names=np.asarray(names))
    report = {'id': '000798', 'output_sha256': sha(out), 'windows_report_sha256': sha(RUN / 'report.json'),
              'windows_audit_sha256': sha(RUN / 'audit_report.json'), 'rows_sha256': result['rows_sha256'],
              'human_report_sha256': sha(HUMAN / 'human_report.json'), 'source_sha256': human['source_sha256'],
              'caption_member_sha256': human['caption_member_sha256'], 'script_sha256': sha(Path(__file__)),
              'urdf_sha256': sha(URDF), 'official_split': human['official_split'],
              'source_group_candidate': human['source_parent'], 'excluded_windows_0_based': [0, 30],
              'raw_frames': [int(frames[0]), int(frames[-1])], 'filter_input_frames': [126, 1014],
              'frames_20hz': grid.tolist(), 'features_shape': list(features.shape), 'decode_max_error_m': err,
              'joint_limit_violation_blended_rad': violation(merged['blended'][:, 7:]),
              'joint_limit_violation_filtered_rad': violation(filtered[:, 7:]),
              'original_caption_records': human['original_caption_records'], 'caption_accepted': False,
              'training_ready': False, 'limitations': ['Experimental cropped window blend and FIR, not author method.',
              'Caption 0/0 refers to original whole segment; cropped semantic coverage is unverified.',
              'Source identity, actual FPS, body-model equivalence and physical quality are unverified.']}
    with (HERE / 'report.json').open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

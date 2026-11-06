"""Pointwise wrist and fidelity constraint search for two rejected Pick frames."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

S = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105')
V2 = S / 'pick_wrist_initialization_diagnostic_v2/candidate_10000007'
V1 = S / 'pick_table_initialization_diagnostic_v1/candidate_10000007'
OUT = S / 'pick_wrist_joint_feasibility_v2.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.is_file():
        print(OUT.read_text())
        return
    plan = json.loads((S / 'generation_pipeline_v5_launch/config.json').read_text())
    report_path = V2 / 'report.json'
    assert sha(report_path) == 'dff8f21f5917f183e7275a0a0c5579ecb0ef8c6e14637c8a917081f7be429c4f'
    report = json.loads(report_path.read_text())
    binding = plan['sources']['postopt']
    assert sha(Path(binding['path'])) == binding['sha256']
    sys.path.insert(0, str(Path(binding['path']).parent))
    from postopt_v2 import load_sample

    spec = importlib.util.spec_from_file_location('diagnostic_fk', plan['fk']['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    urdf = Path(plan['urdf']['path'])
    assert sha(urdf) == plan['urdf']['sha256']
    assert sha(Path(plan['fk']['path'])) == plan['fk']['sha256']
    fk = module.URDFFK(urdf)
    names = fk.actuated
    sample = load_sample(report['source_sample'], report['source_sample_sha256'])
    signature = hashlib.sha256(json.dumps(names, separators=(',', ':')).encode()).hexdigest()
    assert signature == sample['metadata']['coordinate_provenance']['joint_order_sha256']
    tree = ET.parse(urdf).getroot()
    bounds = {joint.get('name'): [float(joint.find('limit').get(k)) for k in ('lower','upper')]
              for joint in tree.findall('joint') if joint.get('name') in names}
    selected = [i for i, name in enumerate(names) if name.startswith(('right_shoulder','right_elbow','right_wrist','waist'))]
    limits = np.array([bounds[names[i]] for i in selected])
    low, high = limits[:, 0] + 1e-6, limits[:, 1] - 1e-6
    targets = sample['targets_world'][:, 21]
    target_r = Rotation.from_quat(sample['orientations'][:, 1, [1,2,3,0]])
    mask = sample['wrist_mask'][:, 1]
    a_path = V2 / 'stage_a_decoded_realization.npz'
    b_path = V2 / 'stage_b_03_raw.npz'
    assert sha(a_path) == report['stage_a_sha256'] and sha(b_path) == report['stages'][2]['sha256']
    with np.load(a_path, allow_pickle=False) as z:
        qa, ra = z['q'], z['root']
    with np.load(b_path, allow_pickle=False) as z:
        qb, rb = z['q'], z['root']
    with np.load(V1 / 'stage_a_decoded_realization.npz', allow_pickle=False) as z:
        q_early = z['q']
    qa = np.clip(qa, [bounds[n][0] + 1e-6 for n in names], [bounds[n][1] - 1e-6 for n in names])
    pose_a = fk.forward(ra[:, 4:], ra[:, [1,2,3,0]], qa, names)
    points, rotation = fk.forward(rb[:, 4:], rb[:, [1,2,3,0]], qb, names, return_rotations=True)
    wrist_pos = np.linalg.norm(points[:, 21] - targets, axis=-1)
    wrist_ang = (target_r.inv() * Rotation.from_matrix(rotation[:, 21])).magnitude()
    peak_pos = int(np.argmax(np.where(mask, wrist_pos, -1)))
    peak_ang = int(np.argmax(np.where(mask, wrist_ang, -1)))
    grip = int(sample['metadata']['grasp_frame'])
    frames = sorted({peak_pos, peak_ang})
    rows = []

    for frame in frames:
        def measured(x):
            angles = qa[frame].copy()
            angles[selected] = x[:len(selected)]
            position = ra[frame, 4:] + x[len(selected):]
            return fk.forward(position[None], ra[frame, [1,2,3,0]][None],
                              angles[None], names, return_rotations=True)

        target = targets[frame]
        orient = target_r[frame]

        def residual(x):
            pp, rr = measured(x)
            pp, rr = pp[0], rr[0]
            angle = (orient.inv() * Rotation.from_matrix(rr[21])).as_rotvec()
            # Use positive residuals only for violated necessary gates, with a
            # small margin below acceptance for numerical precision.
            peak = np.linalg.norm(pp - sample['decoded_world'][frame], axis=-1)
            nonwrist = np.linalg.norm((pp - pose_a[frame])[:20], axis=-1)
            return np.r_[np.maximum(np.linalg.norm(pp[21]-target)/.03-.995,0),
                         np.maximum(np.linalg.norm(angle)/.20-.995,0),
                         np.maximum(peak/.15-.995,0),
                         np.maximum(nonwrist/.08-.995,0),
                         np.maximum(np.linalg.norm(x[len(selected):])/.05-.995,0),
                         (x[:len(selected)]-qa[frame,selected])*.0001]

        anchor = qa[frame, selected]
        for mode, lo, hi, root_radius in (
            ('stage_b_joint_bound_fixed_root', np.maximum(low,anchor-.50), np.minimum(high,anchor+.50), 1e-8),
            ('stage_b_joint_bound_movable_root_xyz', np.maximum(low,anchor-.50), np.minimum(high,anchor+.50), .05),
            ('full_joint_limits_movable_root_xyz', low, high, .05),
        ):
            lower, upper = np.r_[lo, [-root_radius]*3], np.r_[hi, [root_radius]*3]
            trials = []
            for seed in (anchor, qb[frame,selected], q_early[frame,selected]):
                x0 = np.clip(np.r_[seed, np.zeros(3)], lower + 1e-10, upper - 1e-10)
                sol = least_squares(residual, x0, bounds=(lower,upper), max_nfev=150, ftol=1e-7, xtol=1e-7)
                p, r = measured(sol.x)
                p, r = p[0], r[0]
                pe = float(np.linalg.norm(p[21] - target))
                ae = float((orient.inv() * Rotation.from_matrix(r[21])).magnitude())
                rows_metrics = {
                    'position_m':pe, 'orientation_rad':ae, 'both_wrist_gates':pe<.03 and ae<.20,
                    'max_norm_to_wrist_threshold':max(pe/.03, ae/.20),
                    'stage_a_nonwrist_max_m':float(np.linalg.norm((p-pose_a[frame])[:20],axis=-1).max()),
                    'decoded_keypoint_peak_m':float(np.linalg.norm(p-sample['decoded_world'][frame],axis=-1).max()),
                    'stage_a_joint_max_rad':float(np.abs(sol.x[:len(selected)]-anchor).max()),
                    'stage_a_root_shift_m':float(np.linalg.norm(sol.x[len(selected):])),
                    'saturated_joints':[names[selected[i]] for i in range(len(selected))
                                        if min(sol.x[i]-lo[i],hi[i]-sol.x[i])<1e-4],
                    'nfev':int(sol.nfev),
                }
                limits_ratio = max(rows_metrics['max_norm_to_wrist_threshold'], rows_metrics['stage_a_nonwrist_max_m']/.08, rows_metrics['decoded_keypoint_peak_m']/.15, rows_metrics['stage_a_root_shift_m']/.05)
                rows_metrics.update(max_normalized_constraint=limits_ratio, pointwise_subset_feasible=limits_ratio < 1.)
                trials.append(rows_metrics)
            rows.append({'frame':frame, 'mode':mode, 'best':min(trials,key=lambda r:r['max_normalized_constraint'])})
    result = {'status':'pointwise_ik_complete_not_an_admission', 'candidate_id':10000007,
              'raw_sha256':report['source_sample_sha256'], 'wrist_diagnostic_report_sha256':sha(report_path),
              'frames':frames, 'peak_position_frame':peak_pos, 'peak_orientation_frame':peak_ang,
              'source_stage_b':{'max_position_m':float(wrist_pos[peak_pos]),'max_orientation_rad':float(wrist_ang[peak_ang])},
              'joint_names':[names[i] for i in selected], 'scope':'CPU fixed-root-orientation two-frame search for necessary wrist, per-keypoint and root bounds; no proof of infeasibility, collision, tracking, or continuity',
              'rows':rows}
    text = json.dumps(result,indent=2,allow_nan=False)
    with OUT.open('x') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()

"""Check two Pick frames with all 29 joints and root SE(3) under original pointwise bounds."""
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
SOURCE = S / 'pick_wrist_initialization_diagnostic_v2/candidate_10000007'
OUT = S / 'pick_wrist_fullbody_feasibility_v3.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.is_file():
        print(OUT.read_text())
        return
    prior = S / 'pick_wrist_joint_feasibility_v2.json'
    assert sha(prior) == '7d3919029f9259e02f1559597c29e40d3de708e2c98c80798a8283e6c7f8618e'
    plan_path = S / 'generation_pipeline_v5_launch/config.json'
    assert sha(plan_path) == '5d2cfca7b6da4cc9f6b47fb36c63db05ddb6cf8f26244a8189f2486b7a81403a'
    plan = json.loads(plan_path.read_text())
    report_path = SOURCE / 'report.json'
    assert sha(report_path) == 'dff8f21f5917f183e7275a0a0c5579ecb0ef8c6e14637c8a917081f7be429c4f'
    report = json.loads(report_path.read_text())
    binding = plan['sources']['postopt']
    assert sha(Path(binding['path'])) == binding['sha256']
    sys.path.insert(0, str(Path(binding['path']).parent))
    from postopt_v2 import load_sample

    fk_path = Path(plan['fk']['path'])
    assert sha(fk_path) == plan['fk']['sha256']
    spec = importlib.util.spec_from_file_location('diagnostic_fk', fk_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    urdf = Path(plan['urdf']['path'])
    assert sha(urdf) == plan['urdf']['sha256']
    fk = module.URDFFK(urdf)
    names = fk.actuated
    sample = load_sample(report['source_sample'], report['source_sample_sha256'])
    assert hashlib.sha256(json.dumps(names, separators=(',', ':')).encode()).hexdigest() == sample['metadata']['coordinate_provenance']['joint_order_sha256']
    joints = {joint.get('name'): joint.find('limit') for joint in ET.parse(urdf).getroot().findall('joint')
              if joint.get('name') in names}
    low = np.array([float(joints[n].get('lower')) + 1e-6 for n in names])
    high = np.array([float(joints[n].get('upper')) - 1e-6 for n in names])
    a_path, b_path = (SOURCE / name for name in ('stage_a_decoded_realization.npz', 'stage_b_03_raw.npz'))
    assert sha(a_path) == report['stage_a_sha256'] and sha(b_path) == report['stages'][2]['sha256']
    with np.load(a_path, allow_pickle=False) as z:
        qa, ra = np.clip(z['q'], low, high), z['root']
    with np.load(b_path, allow_pickle=False) as z:
        qb, rb = np.clip(z['q'], low, high), z['root']
    pa = fk.forward(ra[:, 4:], ra[:, [1, 2, 3, 0]], qa, names)
    root_a = Rotation.from_quat(ra[:, [1, 2, 3, 0]])
    root_b = Rotation.from_quat(rb[:, [1, 2, 3, 0]])
    target_r = Rotation.from_quat(sample['orientations'][:, 1, [1, 2, 3, 0]])
    frames = sorted({r['frame'] for r in json.loads(prior.read_text())['rows']})
    assert frames == [104,119]
    receipts = []
    for frame in frames:
        assert sample['wrist_mask'][frame,1]
        anchor = qa[frame]
        goal = sample['targets_world'][frame,21]
        reference = sample['decoded_world'][frame]
        lower = np.r_[np.maximum(low, anchor-.50), [-.05]*3, [-np.pi]*3]
        upper = np.r_[np.minimum(high, anchor+.50), [.05]*3, [np.pi]*3]

        def normalized(x):
            state = np.atleast_2d(x)
            orient = (Rotation.from_rotvec(state[:,32:35]) * root_a[frame]).as_quat()
            points, rots = fk.forward(ra[frame,4:] + state[:,29:32], orient, state[:,:29], names,
                                      return_rotations=True)
            angle = (target_r[frame].inv() * Rotation.from_matrix(rots[:,21])).magnitude()
            return np.column_stack((
                np.linalg.norm(points[:,21] - goal, axis=1) / .03,
                angle / .20,
                np.linalg.norm((points-pa[frame])[:,:20], axis=-1) / .08,
                np.linalg.norm(points-reference, axis=-1) / .15,
                np.linalg.norm(state[:,29:32], axis=1) / .05,
            ))

        def residual_many(x):
            state = np.atleast_2d(x)
            return np.column_stack((np.maximum(normalized(state)-.998, 0.), (state[:,:29]-anchor)*.0001))

        def residual(x):
            return residual_many(x)[0]

        def jac(x):
            delta = np.eye(len(x)) * 1e-5
            return ((residual_many(x+delta) - residual_many(x-delta)) / 2e-5).T

        candidates = [
            np.r_[anchor, np.zeros(6)],
            np.r_[qb[frame], rb[frame,4:]-ra[frame,4:], (root_b[frame]*root_a[frame].inv()).as_rotvec()],
        ]
        best = None
        for seed in candidates:
            result = least_squares(residual, np.clip(seed, lower+1e-10, upper-1e-10), jac=jac,
                                   bounds=(lower, upper), max_nfev=240, ftol=1e-8, xtol=1e-8)
            x = result.x
            values = normalized(x)[0]
            result_row = {'frame':frame, 'position_m':values[0]*.03, 'orientation_rad':values[1]*.20,
                          'stage_a_nonwrist_max_m':values[2:22].max()*.08,
                          'decoded_keypoint_peak_m':values[22:44].max()*.15,
                          'stage_a_root_shift_m':values[44]*.05, 'stage_a_joint_max_rad':np.max(np.abs(x[:29]-anchor)),
                          'max_normalized_constraint':values.max(), 'pointwise_subset_feasible':bool(values.max() < 1),
                          'saturated_joints':[names[i] for i in range(29) if min(x[i]-lower[i],upper[i]-x[i]) < 1e-4],
                          'nfev':result.nfev}
            if best is None or result_row['max_normalized_constraint'] < best['max_normalized_constraint']:
                best = result_row
        receipts.append(best)
    result = {'status':'pointwise_fullbody_search_complete_not_an_admission', 'candidate_id':10000007,
              'raw_sha256':report['source_sample_sha256'], 'previous_report_sha256':sha(prior),
              'scope':'Two CPU frames, all 29 joints and root translation/rotation. No collision, tracking or continuity test. Failure is not a global infeasibility proof.',
              'frames':receipts}
    content = json.dumps(result, indent=2, allow_nan=False)
    with OUT.open('x') as stream:
        stream.write(content)
    print(content)


if __name__ == '__main__':
    main()

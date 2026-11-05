"""Verify eight constant-shift variants against original poses and MuJoCo."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT = ROOT / 'eight_projected_preflight_20261031'
MEASURE = ROOT / 'eight_static_robot_quality_20261031'
SCENE = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description/scene_full.xml')
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mid(m, kind, name):
    value = mujoco.mj_name2id(m, kind, name)
    if value < 0:
        raise ValueError(name)
    return value


def main():
    report = json.loads((HERE / 'report.json').read_text())
    base = json.loads((MEASURE / 'report.json').read_text())
    assert sha(MEASURE / 'report.json') == report['floor_diagnostic_sha256']
    assert sha(MEASURE / 'frames.jsonl') == base['frames_sha256'] and sha(SCENE) == base['scene_sha256']
    assert sha(INPUT / 'manifest.json') == report['parent_manifest_sha256']
    assert sha(INPUT / 'state_manifest.json') == report['parent_state_manifest_sha256']
    for key, expected in report['module_sha256'].items():
        assert sha(HERE / key) == expected
    for key in ('manifest', 'state_manifest', 'stats'):
        assert sha(HERE / (key + ('.npz' if key == 'stats' else '.json'))) == report[key + '_sha256']
    old_records = json.loads((INPUT / 'manifest.json').read_text())['records']
    old_states = {x['sample_id']: x for x in json.loads((INPUT / 'state_manifest.json').read_text())['records']}
    new_records = json.loads((HERE / 'manifest.json').read_text())['records']
    new_states = {x['sample_id']: x for x in json.loads((HERE / 'state_manifest.json').read_text())['records']}
    by_id = {x['sample_id']: x for x in report['cases']}
    frames = {}
    for line in (MEASURE / 'frames.jsonl').read_text().splitlines():
        row = json.loads(line)
        frames.setdefault(row['sample_id'], []).append(row)
    model = mujoco.MjModel.from_xml_path(str(SCENE)); data = mujoco.MjData(model)
    floor = mid(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    root = mid(model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_base_joint')
    root_addr = int(model.jnt_qposadr[root]); total = 0; floor_contact_frames = 0; self_count_changes = 0
    worst_gap_error = 0.; worst_translation_error = 0.; results = []; trains = []
    smallest_floor_contact_distance = 0.; floor_frames_below_minus_1e6 = 0
    for old, new in zip(old_records, new_records):
        ident = old['sample_id']; before = old_states[ident]; after = new_states[ident]; case = by_id[ident]
        assert new['sample_id'] == ident and new['parent_feature_sha256'] == old['feature_sha256']
        assert after['parent_state_sha256'] == before['state_sha256']
        assert sha(INPUT / old['feature_path']) == old['feature_sha256'] and sha(HERE / new['feature_path']) == new['feature_sha256']
        assert sha(INPUT / before['state_path']) == before['state_sha256'] and sha(HERE / after['state_path']) == after['state_sha256']
        with np.load(INPUT / before['state_path'], allow_pickle=False) as f:
            root_old, q_old, t_old, names_old = (f[x].copy() for x in ('root_wxyz_xyz', 'joint_pos', 'sample_times', 'joint_names'))
        with np.load(HERE / after['state_path'], allow_pickle=False) as f:
            roots, q, times, names = (f[x].copy() for x in ('root_wxyz_xyz', 'joint_pos', 'sample_times', 'joint_names'))
        assert np.array_equal(q, q_old) and np.array_equal(times, t_old) and np.array_equal(names, names_old)
        change = roots.astype(float) - root_old.astype(float)
        reference = np.zeros_like(change); reference[:, 6] = case['root_shift_z_m']
        err = float(np.abs(change - reference).max()); worst_translation_error = max(worst_translation_error, err)
        assert err < 1e-6
        with np.load(INPUT / old['feature_path'], allow_pickle=False) as f:
            old_feature = f['features'].copy()
        with np.load(HERE / new['feature_path'], allow_pickle=False) as f:
            new_feature = f['features'].copy()
        delta = float(np.abs(new_feature.astype(float) - old_feature.astype(float)).max())
        assert abs(delta - case['feature_max_abs_change']) < 1e-12
        assert int(np.count_nonzero(new_feature[:, -4:] != old_feature[:, -4:])) == case['foot_proxy_values_changed']
        if new['split'] == 'train':
            trains.append(new_feature.astype(np.float64))
        ids = [int(model.jnt_qposadr[mid(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))]) for name in names]
        mins = [float('inf'), float('inf')]; num_floor = 0; changed_self = 0
        for i in range(len(times)):
            mujoco.mj_resetData(model, data); data.qpos[root_addr:root_addr+3] = roots[i, 4:]
            data.qpos[root_addr+3:root_addr+7] = roots[i, :4]
            for address, value in zip(ids, q[i]): data.qpos[address] = value
            mujoco.mj_forward(model, data)
            normal = data.geom_xmat[floor].reshape(3, 3)[:, 2]; origin = data.geom_xpos[floor]
            for side, name in enumerate(('left_ankle_roll_link', 'right_ankle_roll_link')):
                body = mid(model, mujoco.mjtObj.mjOBJ_BODY, name)
                gap = min(float((data.geom_xpos[g] - origin) @ normal - model.geom_size[g, 0]) for g in range(model.ngeom)
                          if model.geom_bodyid[g] == body and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and
                          (model.geom_contype[g] or model.geom_conaffinity[g]))
                mins[side] = min(mins[side], gap)
                predicted = frames[ident][i]['foot_sphere_min_gaps_m'][side] + float(change[i, 6])
                gap_error = abs(predicted - gap); worst_gap_error = max(worst_gap_error, gap_error)
                assert gap_error < 1e-8
            floor_dists = [float(data.contact[j].dist) for j in range(data.ncon) if data.contact[j].dist < 0 and floor in
                           (int(data.contact[j].geom1), int(data.contact[j].geom2))]
            floor_n = len(floor_dists)
            if floor_dists:
                smallest_floor_contact_distance = min(smallest_floor_contact_distance, min(floor_dists))
                floor_frames_below_minus_1e6 += min(floor_dists) < -1e-6
            self_n = sum(1 for j in range(data.ncon) if data.contact[j].dist < 0 and floor not in
                         (int(data.contact[j].geom1), int(data.contact[j].geom2)))
            num_floor += bool(floor_n)
            changed_self += self_n != len(frames[ident][i]['self_penetration_contacts'])
        results.append({'sample_id': ident, 'shift_z_m': case['root_shift_z_m'], 'new_foot_min_gap_m': mins,
                        'floor_penetration_contact_frames': num_floor, 'self_contact_count_changed_frames': changed_self,
                        'max_feature_component_change': delta})
        total += len(times); floor_contact_frames += num_floor; self_count_changes += changed_self
    train = np.concatenate(trains)
    with np.load(HERE / 'stats.npz', allow_pickle=False) as f:
        meanerr = float(np.abs(f['feature_mean'] - train.mean(0)).max())
        stderr = float(np.abs(f['feature_std'] - np.maximum(train.std(0), .001)).max())
    assert len(train) == report['train_frames'] == 296 and max(meanerr, stderr) < 1e-6
    out = {'report_sha256': sha(HERE / 'report.json'), 'cases': results, 'frames_recomputed': total,
           'foot_gap_prediction_max_error_m': worst_gap_error, 'translation_max_error_m': worst_translation_error,
           'floor_penetration_contact_frames': floor_contact_frames, 'self_contact_count_changed_frames': self_count_changes,
           'minimum_floor_contact_distance_m': smallest_floor_contact_distance,
           'floor_contact_frames_below_minus_1e6_m': int(floor_frames_below_minus_1e6),
           'train_feature_mean_max_error': meanerr, 'train_feature_std_max_error': stderr, 'training_ready': False,
           'limitations': ['Same MuJoCo engine and assumed z=0 plane; not true ground or measured contact.',
                           'No independent full FK/encoding, physical dynamics, caption or source acceptance.']}
    with (HERE / 'audit_report_v2.json').open('x', encoding='utf8') as f:
        json.dump(out, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()

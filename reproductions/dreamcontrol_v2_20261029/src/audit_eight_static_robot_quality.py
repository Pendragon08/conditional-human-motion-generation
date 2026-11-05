"""Recount all static records and rerun three poses per trajectory in MuJoCo."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT = ROOT / 'eight_projected_preflight_20261031'
BASE = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description')
SCENE = BASE / 'scene_full.xml'
XML = BASE / 'g1_27dof_with_hand_rev_1_0.xml'
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def obj(m, kind, name):
    result = mujoco.mj_name2id(m, kind, name)
    if result < 0:
        raise ValueError(name)
    return result


def main():
    report = json.loads((HERE / 'report.json').read_text())
    assert sha(SCENE) == report['scene_sha256'] and sha(XML) == report['model_xml_sha256']
    assert sha(INPUT / 'manifest.json') == report['manifest_sha256']
    assert sha(INPUT / 'state_manifest.json') == report['state_manifest_sha256']
    assert sha(HERE / 'frames.jsonl') == report['frames_sha256']
    for entry in report['model_meshes']:
        assert sha(Path(entry['path'])) == entry['sha256']
    frames = [json.loads(line) for line in (HERE / 'frames.jsonl').read_text().splitlines()]
    rows = json.loads((INPUT / 'manifest.json').read_text())['records']
    states = {x['sample_id']: x for x in json.loads((INPUT / 'state_manifest.json').read_text())['records']}
    model = mujoco.MjModel.from_xml_path(str(SCENE)); data = mujoco.MjData(model)
    floor = obj(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    root = obj(model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_base_joint')
    adr = int(model.jnt_qposadr[root])
    checked = 0
    max_gap_error = 0.
    for record, summary in zip(rows, report['cases']):
        ident = record['sample_id']; state = states[ident]
        assert summary['sample_id'] == ident and sha(INPUT / state['state_path']) == state['state_sha256']
        selected = [x for x in frames if x['sample_id'] == ident]
        assert len(selected) == summary['frames']
        assert [x['sample_index'] for x in selected] == list(range(len(selected)))
        gaps = np.array([x['foot_sphere_min_gaps_m'] for x in selected])
        np.testing.assert_allclose(gaps.min(0), summary['foot_gap_min_m'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(gaps.max(0), summary['foot_gap_max_m'], atol=1e-12, rtol=0)
        assert [(gaps[:, j] < 0).sum() for j in range(2)] == summary['foot_below_assumed_plane_frames']
        assert sum(bool(x['floor_penetration_contacts']) for x in selected) == summary['frames_with_floor_penetration_contact']
        assert sum(bool(x['self_penetration_contacts']) for x in selected) == summary['frames_with_self_penetration_contact']
        with np.load(INPUT / state['state_path'], allow_pickle=False) as f:
            roots, q, t = f['root_wxyz_xyz'].copy(), f['joint_pos'].copy(), f['sample_times'].copy()
            names = [str(x) for x in f['joint_names']]
        rates = np.diff(q, axis=0) / np.diff(t)[:, None]
        assert abs(float(np.abs(rates).max()) - summary['max_abs_joint_speed_rad_s']) < 1e-7
        joint_addr = [int(model.jnt_qposadr[obj(model, mujoco.mjtObj.mjOBJ_JOINT, x)]) for x in names]
        for i in sorted(set([0, len(t)//2, len(t)-1])):
            mujoco.mj_resetData(model, data)
            data.qpos[adr:adr + 3] = roots[i, 4:]
            data.qpos[adr + 3:adr + 7] = roots[i, :4]
            for addr, val in zip(joint_addr, q[i]):
                data.qpos[addr] = val
            mujoco.mj_forward(model, data)
            floor_pos = np.array(data.geom_xpos[floor])
            normal = np.array(data.geom_xmat[floor]).reshape(3, 3)[:, 2]
            for side, name in enumerate(('left_ankle_roll_link', 'right_ankle_roll_link')):
                body = obj(model, mujoco.mjtObj.mjOBJ_BODY, name)
                values = []
                for g in range(model.ngeom):
                    if (model.geom_bodyid[g] == body and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and
                        (model.geom_contype[g] or model.geom_conaffinity[g])):
                        distance = float(sum((data.geom_xpos[g, a] - floor_pos[a]) * normal[a] for a in range(3)) - model.geom_size[g, 0])
                        values.append(distance)
                assert len(values) == 4
                error = abs(min(values) - selected[i]['foot_sphere_min_gaps_m'][side])
                max_gap_error = max(max_gap_error, error)
                assert error < 1e-10
            actual_floor = sum(1 for j in range(data.ncon) if data.contact[j].dist < 0 and floor in
                               (int(data.contact[j].geom1), int(data.contact[j].geom2)))
            actual_self = sum(1 for j in range(data.ncon) if data.contact[j].dist < 0 and floor not in
                              (int(data.contact[j].geom1), int(data.contact[j].geom2)))
            assert actual_floor == len(selected[i]['floor_penetration_contacts'])
            assert actual_self == len(selected[i]['self_penetration_contacts'])
            checked += 1
    assert len(frames) == sum(x['frames'] for x in report['cases'])
    out = {'report_sha256': sha(HERE / 'report.json'), 'frames_sha256': sha(HERE / 'frames.jsonl'),
           'cases': len(rows), 'total_frames_recounted': len(frames), 'poses_recomputed': checked,
           'max_sphere_gap_error_m': max_gap_error, 'mujoco_version': mujoco.__version__, 'training_ready': False,
           'limitations': 'Same MuJoCo model/solver, not independent collision engine, true ground, dynamics or physical acceptance.'}
    with (HERE / 'audit_report.json').open('x', encoding='utf8') as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()

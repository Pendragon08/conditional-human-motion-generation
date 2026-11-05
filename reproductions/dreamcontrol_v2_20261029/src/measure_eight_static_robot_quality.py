"""Measure all eight G1 trajectories against a fixed MuJoCo scene.

Kinematic poses only, open-hand defaults, assumed plane and MuJoCo collision
filters; no measured contact, balance, force, PhysX equivalence or automatic
training verdict. Magnitudes and contacts are reported without paper claims.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT = ROOT / 'eight_projected_preflight_20261031'
BASE = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description')
SCENE = BASE / 'scene_full.xml'
MODEL_XML = BASE / 'g1_27dof_with_hand_rev_1_0.xml'
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mid(model, kind, name):
    i = mujoco.mj_name2id(model, kind, name)
    if i < 0:
        raise ValueError('missing MuJoCo name: ' + name)
    return i


def asset_hashes():
    tree = ET.parse(MODEL_XML)
    paths = sorted({(BASE / 'meshes' / m.get('file')).resolve(strict=True)
                    for m in tree.findall('.//asset/mesh') if m.get('file')})
    if not paths or any(not p.is_relative_to(BASE) for p in paths):
        raise ValueError('unexpected model mesh paths')
    return [{'path': str(p), 'sha256': sha(p)} for p in paths]


def main():
    build = json.loads((INPUT / 'build_report.json').read_text())
    manifest, evidence = INPUT / 'manifest.json', INPUT / 'state_manifest.json'
    if sha(manifest) != build['manifest_sha256'] or sha(evidence) != build['state_manifest_sha256']:
        raise ValueError('manifest mismatch')
    records = json.loads(manifest.read_text())['records']
    states = {x['sample_id']: x for x in json.loads(evidence.read_text())['records']}
    if len(records) != 8 or len(states) != 8:
        raise ValueError('unexpected record count')
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    root = mid(model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_base_joint')
    if model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('expected floating-base joint')
    rootadr = int(model.jnt_qposadr[root])
    floor = mid(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    feet = [mid(model, mujoco.mjtObj.mjOBJ_BODY, side + '_ankle_roll_link') for side in ('left', 'right')]
    spheres = [[g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body and
                int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_SPHERE) and
                (model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0)] for body in feet]
    if any(len(group) != 4 for group in spheres):
        raise ValueError('expected 4 collision spheres per foot')
    meshlist = asset_hashes()
    cases = []
    frame_path = HERE / 'frames.jsonl'
    with frame_path.open('x', encoding='utf8', buffering=1) as stream:
        for record in records:
            state = states[record['sample_id']]
            path = INPUT / state['state_path']
            if sha(path) != state['state_sha256'] or state['feature_sha256'] != record['feature_sha256']:
                raise ValueError('state provenance mismatch')
            with np.load(path, allow_pickle=False) as f:
                roots = f['root_wxyz_xyz'].astype(float)
                joints = f['joint_pos'].astype(float)
                times = f['sample_times'].astype(float)
                names = [str(n) for n in f['joint_names']]
            n = len(times)
            if (roots.shape != (n, 7) or joints.shape != (n, 27) or len(set(names)) != 27 or n < 2 or
                not np.isfinite(roots).all() or not np.isfinite(joints).all() or
                not np.allclose(np.diff(times), .05, rtol=0, atol=1e-8) or
                not np.allclose(np.linalg.norm(roots[:, :4], axis=1), 1, rtol=0, atol=1e-5)):
                raise ValueError('invalid state or timeline')
            addresses = [int(model.jnt_qposadr[mid(model, mujoco.mjtObj.mjOBJ_JOINT, name)]) for name in names]
            samples, floor_penetration, self_penetration = [], 0, 0
            worst_self = None
            for i in range(n):
                mujoco.mj_resetData(model, data)
                data.qpos[rootadr:rootadr + 3] = roots[i, 4:]
                data.qpos[rootadr + 3:rootadr + 7] = roots[i, :4]
                data.qpos[addresses] = joints[i]
                mujoco.mj_forward(model, data)
                normal = data.geom_xmat[floor].reshape(3, 3)[:, 2]
                origin = data.geom_xpos[floor]
                gaps = [min(float((data.geom_xpos[g] - origin) @ normal - model.geom_size[g, 0]) for g in group) for group in spheres]
                self_contacts, floor_contacts = [], []
                for k in range(data.ncon):
                    c = data.contact[k]
                    if c.dist >= 0:
                        continue
                    a, b = int(c.geom1), int(c.geom2)
                    def body(g):
                        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or 'unnamed'
                    item = {'bodies': [body(a), body(b)], 'dist_m': float(c.dist)}
                    (floor_contacts if floor in (a, b) else self_contacts).append(item)
                floor_penetration += bool(floor_contacts)
                self_penetration += bool(self_contacts)
                for item in self_contacts:
                    if worst_self is None or item['dist_m'] < worst_self['dist_m']:
                        worst_self = {'sample_index': i, **item}
                row = {'sample_id': record['sample_id'], 'sample_index': i, 'nominal_time_s': float(times[i]),
                       'foot_sphere_min_gaps_m': gaps, 'floor_penetration_contacts': floor_contacts,
                       'self_penetration_contacts': self_contacts}
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                samples.append(gaps)
            gaps = np.asarray(samples)
            velocity = np.diff(joints, axis=0) / np.diff(times)[:, None]
            root_speed = np.linalg.norm(np.diff(roots[:, 4:], axis=0), axis=1) / np.diff(times)
            cases.append({'sample_id': record['sample_id'], 'split': record['split'], 'frames': n,
                          'state_sha256': state['state_sha256'], 'feature_sha256': record['feature_sha256'],
                          'foot_gap_min_m': gaps.min(axis=0).tolist(), 'foot_gap_max_m': gaps.max(axis=0).tolist(),
                          'foot_below_assumed_plane_frames': (gaps < 0).sum(axis=0).tolist(),
                          'frames_with_floor_penetration_contact': floor_penetration,
                          'frames_with_self_penetration_contact': self_penetration, 'worst_self_contact': worst_self,
                          'max_abs_joint_speed_rad_s': float(np.abs(velocity).max()),
                          'max_root_speed_m_s': float(root_speed.max())})
    out = {'manifest_sha256': sha(manifest), 'state_manifest_sha256': sha(evidence),
           'scene_sha256': sha(SCENE), 'model_xml_sha256': sha(MODEL_XML), 'model_meshes': meshlist,
           'mujoco_version': mujoco.__version__, 'script_sha256': sha(Path(__file__)),
           'frames_sha256': sha(frame_path), 'floor_position': data.geom_xpos[floor].tolist(),
           'floor_normal': data.geom_xmat[floor].reshape(3, 3)[:, 2].tolist(),
           'foot_spheres_per_side': [len(x) for x in spheres], 'cases': cases, 'training_ready': False,
           'limitations': ['Static MuJoCo kinematics only, hands in default state, no integration or contact forces.',
                           'Assumed z=0 plane; source ground and true timing are not independently established.',
                           'Contacts reflect this model and its collision filters; geometry differs from Isaac USD.',
                           'No automatically selected rejection threshold or author feasibility claim.',
                           'No source, caption, physics or training acceptance.']}
    with (HERE / 'report.json').open('x', encoding='utf8') as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'cases': len(cases), 'frames': sum(x['frames'] for x in cases), 'training_ready': False}))


if __name__ == '__main__':
    main()

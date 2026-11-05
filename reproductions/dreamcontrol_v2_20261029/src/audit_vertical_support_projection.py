"""Recompute corrected COM and vertical bound; no quality acceptance implied."""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT = ROOT / 'four_semantic_subset_20261101'
DATA = ROOT / 'six_family_disjoint_preview_v2_20261101'
BOUND = ROOT / 'vertical_support_bound_20261101'
HERE = Path(__file__).resolve().parent
SCENE = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description/scene_full.xml')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    report = json.loads((HERE / 'report.json').read_text())
    bound = json.loads((BOUND / 'report.json').read_text())
    assert sha(BOUND / 'report.json') == report['bound_report_sha256']
    assert sha(SCENE) == bound['scene_sha256']
    assert sha(INPUT / 'manifest.json') == report['input_manifest_sha256']
    states = {x['sample_id']: x for x in json.loads((INPUT / 'state_manifest.json').read_text())['records']}
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(model)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_base_joint')
    assert joint >= 0
    root, adr = int(model.jnt_bodyid[joint]), int(model.jnt_qposadr[joint])
    bodies = [i for i in range(model.nbody) if model.body_rootid[i] == root]
    mass = sum(float(model.body_mass[i]) for i in bodies)
    assert abs(mass-bound['robot_mass_kg']) < 1e-9
    cases = []
    with np.load(BOUND / 'trajectories.npz', allow_pickle=False) as baseline:
        for case in report['cases']:
            sid = case['sample_id']
            if case['status'] != 'projected':
                cases.append({'sample_id': sid, 'status': 'not_projected'})
                continue
            source = DATA / states[sid]['state_path']
            output = HERE / (sid + '.npz')
            assert sha(source) == case['source_state_sha256'] and sha(output) == case['output_sha256']
            with np.load(source, allow_pickle=False) as old, np.load(output, allow_pickle=False) as new:
                original = old['root_wxyz_xyz'].astype(np.float64)
                roots, shift, times = new['root_wxyz_xyz'], new['height_shift_m'], new['sample_times']
                np.testing.assert_array_equal(old['joint_pos'], new['joint_pos'])
                np.testing.assert_array_equal(old['joint_names'], new['joint_names'])
                np.testing.assert_array_equal(old['sample_times'], times)
                np.testing.assert_array_equal(original[:, :6], roots[:, :6])
                np.testing.assert_allclose(roots[:, 6] - original[:, 6], shift, rtol=0, atol=1e-15)
                assert np.max(np.abs(shift)) <= report['max_root_z_shift_m'] + 1e-12
                assert abs(shift[0]) < 1e-12 and abs(shift[-1]) < 1e-12
                addresses = []
                for n in new['joint_names']:
                    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(n))
                    assert j >= 0
                    addresses.append(int(model.jnt_qposadr[j]))
                com = []
                for root_state, angles in zip(roots, new['joint_pos']):
                    mujoco.mj_resetData(model, d)
                    d.qpos[adr:adr+3], d.qpos[adr+3:adr+7] = root_state[4:], root_state[:4]
                    for address, angle in zip(addresses, angles):
                        d.qpos[address] = angle
                    mujoco.mj_forward(model, d)
                    com.append([sum(float(model.body_mass[i])*float(d.xipos[i,k]) for i in bodies)/mass for k in range(3)])
                com = np.asarray(com)
                error = float(np.max(np.abs(com-new['com_xyz'])))
                np.testing.assert_allclose(com, new['com_xyz'], rtol=0, atol=2e-12)
                expected = baseline[sid+'_com'].copy()
                expected[:, 2] += shift
                np.testing.assert_allclose(com, expected, rtol=0, atol=2e-12)
                dt = float(times[1]-times[0])
                force = np.array([(float(com[i+1,2])-2*float(com[i,2])+float(com[i-1,2]))/dt**2-float(model.opt.gravity[2]) for i in range(1,len(times)-1)])
                np.testing.assert_allclose(force, new['support_after'], rtol=0, atol=2e-9)
                assert np.min(force) >= -report['support_numerical_tolerance_n_per_kg']
                assert abs(np.max(np.abs(shift))-case['max_abs_shift_m']) < 1e-12
                cases.append({'sample_id': sid, 'frames_recomputed': len(times), 'max_com_error_m': error,
                              'min_support_n_per_kg': float(force.min()), 'max_abs_shift_m': float(np.max(np.abs(shift)))})
    assert 'Ran 3 tests' in (HERE/'tests.log').read_text() and 'OK' in (HERE/'tests.log').read_text()
    out = {'projection_report_sha256': sha(HERE/'report.json'), 'script_sha256': sha(Path(__file__)),
           'frames_recomputed': sum(x.get('frames_recomputed', 0) for x in cases), 'cases': cases,
           'training_ready': False, 'limitations': 'Uses the same assumed model and sample timing. Tests one necessary inequality, not full dynamics or contact.'}
    with (HERE/'audit.json').open('x') as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(out))

if __name__ == '__main__':
    main()

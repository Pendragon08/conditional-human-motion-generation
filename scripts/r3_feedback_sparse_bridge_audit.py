"""Representation ablation only: robot FK -> sparse human conditions.

This does not fabricate a full HumanML3D prefix or claim diffusion/control
benefit. Calibration observes training references only; test references are
held out. No robot trajectory is edited and no extra IK is used.
"""
import json
from pathlib import Path
import numpy as np
import mujoco
from r3_sim2sim_mujoco import Transfer, sha

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/feedback_world_model_20261006'
OUT = ROOT/'results/feedback_world_model_20261006'
R = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=np.float64)


def report_errors(predicted, truth):
    err = np.linalg.norm(predicted-truth, axis=-1)
    return [{'joint': name, 'mean_error_m': float(err[..., j].mean()),
             'p95_error_m': float(np.quantile(err[..., j], .95))}
            for j,name in enumerate(('pelvis', 'left_ankle', 'right_ankle'))]


def main():
    pairing = json.loads((ART/'sources/mapping.json').read_text())
    assert pairing['status'] == 'PASS'
    human = np.load(ART/'sources/paired_human_xyz.npz')['human_xyz'][:,:, [0,7,8]]
    t = Transfer('Kick', 'cuda:0')
    refs = t.ref.a
    ids = [mujoco.mj_name2id(t.model, mujoco.mjtObj.mjOBJ_BODY, name)
           for name in ('pelvis', 'left_ankle_roll_link', 'right_ankle_roll_link')]
    robot = np.zeros_like(human)
    for i in range(66):
        for f in range(180):
            mujoco.mj_resetData(t.model, t.data)
            t.data.qpos[:3] = refs['transl'][i, f+20]+refs['offsets'][i]
            t.data.qpos[3:7] = refs['quats'][i, f+20]
            t.data.qpos[t.qadr] = refs['dof_pos'][i, f+20]
            mujoco.mj_forward(t.model, t.data)
            robot[i,f] = t.data.xpos[ids]
    # Explicit fixed original axis conversion: row vector robot @ R -> human.
    naive = robot@R
    train, val, test = slice(0,36), slice(36,46), slice(46,66)
    outputs = np.empty_like(human)
    weights = []
    for j in range(3):
        x = (robot[:,:,j] if j == 0 else robot[:,:,j]-robot[:,:,0])@R
        y = human[:,:,j] if j == 0 else human[:,:,j]-human[:,:,0]
        design = np.c_[x[train].reshape(-1,3), np.ones(36*180)]
        w = np.linalg.lstsq(design, y[train].reshape(-1,3), rcond=None)[0]
        outputs[:,:,j] = (np.c_[x.reshape(-1,3), np.ones(66*180)]@w).reshape(66,180,3)
        weights.append(w)
    outputs[:,:,1:] += outputs[:,:,:1]
    # Known initial human plan is available at reset; root-relative origin
    # alignment is a legitimate cheap comparison and must not be omitted.
    origin = np.empty_like(human)
    anchored = np.empty_like(human)
    anchored_weights = []
    for j in range(3):
        if j == 0:
            x = robot[:,:,j]@R
            y = human[:,:,j]
        else:
            x = (robot[:,:,j]-robot[:,:,0])@R
            y = human[:,:,j]-human[:,:,0]
        xd, yd = x-x[:,:1], y-y[:,:1]
        w = np.linalg.lstsq(xd[train].reshape(-1,3), yd[train].reshape(-1,3), rcond=None)[0]
        origin[:,:,j] = y[:,:1]+xd
        anchored[:,:,j] = y[:,:1]+xd@w
        anchored_weights.append(w)
    origin[:,:,1:] += origin[:,:,:1]
    anchored[:,:,1:] += anchored[:,:,:1]
    rows = {}
    for name, selected in (('train',train),('validation',val),('test',test)):
        rows[name] = {'naive_axis_only': report_errors(naive[selected], human[selected]),
                      'calibrated_sparse': report_errors(outputs[selected], human[selected]),
                      'known_initial_origin_alignment': report_errors(origin[selected], human[selected]),
                      'anchored_delta_calibration': report_errors(anchored[selected], human[selected])}
    test_rows = rows['test']['anchored_delta_calibration']
    passed = all(r['mean_error_m'] <= .08 and r['p95_error_m'] <= .15 for r in test_rows)
    np.savez_compressed(ART/'sources/sparse_bridge_v2.npz', weights=np.stack(weights),
                        anchored_delta_weights=np.stack(anchored_weights),
                        joint_indices=[0,7,8], axis_rotation=R)
    # Positive algebraic fixture checks direction and units; it is not a real
    # G1->HumanML round-trip or policy outcome.
    sample = np.array([[.3,.8,.5],[-.2,1.1,.7]])
    fixture_error = np.max(np.abs((sample@R.T)@R-sample))
    assert fixture_error < 1e-12
    result = {'status': 'COMPLETE', 'module': 'sparse_feedback_representation_ablation',
              'paired_references':66, 'train_references':36, 'validation_references':10,
              'test_references':20, 'frames_per_reference':180,
              'thresholds':{'mean_m':.08,'p95_m':.15}, 'sparse_bridge_gate_pass':passed,
              'source_identity_pairing_pass':True, 'axis_algebra_fixture_error':float(fixture_error),
              'exact_motion_pairing':pairing['entries'], 'metrics':rows,
              'calibration_sha256':sha(ART/'sources/sparse_bridge_v2.npz'),
              'source_sha256':pairing['official_human_sha256'], 'no_extra_robot_ik':True,
              'generator_intervention_done':False,'full_humanml_prefix_verified':False,
              'real_execution_mapping_accuracy_verified':False,
              'limitations':['Calibration uses reference FK, not observed human ground truth for an actual G1 execution.',
                 'Only pelvis/ankles are mapped; no orientation, full263D HumanML, contact or whole-body semantics certification.',
                 'Affine map may not extrapolate to fallen/perturbed robot states.',
                 'Passing sparse calibration is insufficient to approve complete CLoSD prefix transfer.']}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'feedback_bridge_v2.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SPARSE_FEEDBACK_BRIDGE_AUDIT',json.dumps({k:v for k,v in result.items() if k!='exact_motion_pairing'}),flush=True)


if __name__ == '__main__':
    main()

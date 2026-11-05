"""Audit a raw-GRAB initialized 12-keypoint refinement, not training data."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.signal import resample_poly
import mujoco

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0, str(ROOT/'foot_contact_fit_20261030'))
sys.path.insert(1, str(ROOT/'antialias_mujoco_20261030'))
from motion_bridge import SOURCE_NAMES
from mujoco_kinematic_contacts import scan
from mujoco_dynamic_contact_witness import simulate
from human_keypoint_robot_fit import fit_human_keypoints


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    pilot = ROOT/'grab_human_robot_pilot_20261030'
    report = json.loads((pilot/'report.json').read_text())
    artifact = pilot/'pilot_arrays.npz'
    if sha(artifact) != report['artifact_sha256']:
        raise ValueError('pilot hash mismatch')
    project=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
    u27=project/'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
    scene=project/'Sim2Real/resources/robots/g1_description/scene_full.xml'
    gains=project/'Sim2Real/deploy/deploy_mujoco/configs/g1_full_body.yaml'
    if sha(u27) != report['urdf27_sha256']:
        raise ValueError('robot asset changed')
    with np.load(artifact, allow_pickle=False) as data:
        human=data['human_joints'].copy()
        names=data['g1_27_joint_names'].tolist()
        root=data['g1_27_root_xyz'].copy(); quat=data['g1_27_root_quat_xyzw'].copy()
        joints=data['g1_27_joint_pos'].copy()
    human20=resample_poly(human, 1, 6, axis=0, padtype='line')[:len(root)]
    human20[0]=human[0]
    if len(human20)!=len(root): raise ValueError('retimed human/robot length mismatch')
    order=[n for n in SOURCE_NAMES if n not in ('waist_roll_joint','waist_pitch_joint')]
    if set(order)!=set(names): raise ValueError('joint name mismatch')
    motion=np.c_[root, quat, joints[:, [names.index(x) for x in order]]]
    fit=fit_human_keypoints(human20, motion, u27)
    model=mujoco.MjModel.from_xml_path(str(scene))
    initial={'root_pos':root,'root_quat_xyzw':quat,'joint_pos':joints,'joint_names':names}
    metrics={'initial_static':scan(model, initial),'refined_static':scan(model, fit),
             'initial_dynamic':simulate(scene,gains,u27,initial,seconds=.9),
             'refined_dynamic':simulate(scene,gains,u27,fit,seconds=.9)}
    out=Path(__file__).with_name('refined_arrays.npz')
    with out.open('xb') as f:
        np.savez_compressed(f, root_pos=fit['root_pos'], root_quat_xyzw=fit['root_quat_xyzw'],
                            joint_pos=fit['joint_pos'], joint_names=np.asarray(fit['joint_names']),
                            target_relative_world=fit['target_relative_world'])
    doc={'pilot_sha256':sha(artifact),'script_sha256':sha(__file__),
         'fit_code_sha256':sha(Path(__file__).with_name('human_keypoint_robot_fit.py')),
         'refined_arrays_sha256':sha(out),'human_relative_keypoint_error':{k:fit[k] for k in
             ('initial_mean_error_m','final_mean_error_m','initial_max_error_m','final_max_error_m','low_speed_foot_intervals','steps')},
         'scene_sha256':sha(scene),'gains_sha256':sha(gains),'metrics':metrics,
         'warning':'Single 0.95s subject-2 airplane segment. Relative 12-joint correspondence, leg/arm scaling and bone-direction loss are explicit approximations; human height is assumed and direct coordinates are not ground truth. Human joints FIR and quaternion filtering are approximations. Source floor unknown; static and simple-PD dynamics use old MuJoCo scene, gravity -7.5, no object or learned balance. Not PyRoki, paper method, genuine ground truth, or training acceptance.'}
    with Path(__file__).with_name('refined_report.json').open('x',encoding='utf8') as f:
        json.dump(doc,f,indent=2,allow_nan=False)
    print('completed human-keypoint comparison',flush=True)


if __name__ == '__main__': main()

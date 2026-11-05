"""Validate a short legacy-PyRoki GRAB output without accepting training data."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import mujoco

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0,str(ROOT/'foot_contact_fit_20261030'))
sys.path.insert(1,str(ROOT/'antialias_mujoco_20261030'))
from g1_fk import URDFFK
from waist_lock_ik import FIT_LINKS, _joint_limits
from mujoco_kinematic_contacts import scan
from mujoco_dynamic_contact_witness import simulate


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    run=ROOT/'pyroki_grab_pilot_20261030'; report=run/'report.json'; artifact=run/'raw_pilot.npz'
    receipt=json.loads(report.read_text())
    if digest(artifact)!=receipt['artifact_sha256']: raise ValueError('output hash mismatch')
    source=ROOT/'grab_human_robot_pilot_20261030/pilot_arrays.npz'
    if digest(source)!=receipt['source_sha256']: raise ValueError('source hash mismatch')
    with np.load(artifact,allow_pickle=False) as data:
        names=data['robot_joint_names'].tolist(); joints=data['robot_joint_pos'].astype(float)
        root=data['robot_root_wxyz_xyz'].astype(float); frames=data['frames'].astype(int)
        human=data['human_keypoints'].astype(float); fps=float(data['fps'])
    with np.load(source,allow_pickle=False) as data:
        source_frames=data['input_frame_indices']; source_human=data['human_joints']
        if fps!=float(data['input_fps']): raise ValueError('source rate mismatch')
        indexes=np.searchsorted(source_frames,frames)
        if np.any(indexes>=len(source_frames)) or not np.array_equal(source_frames[indexes],frames):
            raise ValueError('source frames mismatch')
        if not np.allclose(source_human[indexes],human,atol=0,rtol=0): raise ValueError('human payload mismatch')
    if len(frames)!=7 or not np.all(np.diff(frames)==1) or fps!=120.:
        raise ValueError('unexpected native timeline')
    if joints.shape!=(7,27) or root.shape!=(7,7) or human.shape!=(7,22,3):
        raise ValueError('incorrect output shape')
    if not all(np.isfinite(x).all() for x in (joints,root,human)):
        raise ValueError('nonfinite values')
    quat=root[:,[1,2,3,0]]; norm=np.linalg.norm(quat,axis=1)
    if not np.allclose(norm,1.,atol=1e-4,rtol=0): raise ValueError('invalid root quaternion')
    project=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
    urdf=project/'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
    scene=project/'Sim2Real/resources/robots/g1_description/scene_full.xml'
    gains=project/'Sim2Real/deploy/deploy_mujoco/configs/g1_full_body.yaml'
    if digest(urdf)!=receipt['urdf_sha256']: raise ValueError('robot asset mismatch')
    lower,upper=_joint_limits(urdf,names); low=lower.numpy(); high=upper.numpy()
    violation=np.maximum(np.maximum(low-joints,joints-high),0.)
    fk=URDFFK(urdf).forward(root[:,4:],quat,joints,names,links=FIT_LINKS)
    human_idx=[1,2,4,5,7,8,16,17,18,19,20,21]
    world_error=np.linalg.norm(fk-human[:,human_idx,:],axis=-1)
    fit={'root_pos':root[:,4:],'root_quat_xyzw':quat,'joint_pos':joints,'joint_names':names}
    model=mujoco.MjModel.from_xml_path(str(scene)); static=scan(model,fit)
    # Seven frames span only 0.05 seconds. First and last coincide exactly
    # with the 20 Hz grid; no claim of adequate low-pass filtering is made.
    target_times=np.array([0.,.05]); source_times=(frames-frames[0])/fps
    if not np.isclose(source_times[-1],.05,atol=1e-12): raise ValueError('unexpected duration')
    q20=np.stack([np.interp(target_times,source_times,joints[:,i]) for i in range(27)],axis=1)
    p20=np.stack([np.interp(target_times,source_times,root[:,4+i]) for i in range(3)],axis=1)
    quat20=Slerp(source_times,Rotation.from_quat(quat))(target_times).as_quat()
    short={'root_pos':p20,'root_quat_xyzw':quat20,'joint_pos':q20,'joint_names':names}
    dynamic=simulate(scene,gains,urdf,short,seconds=.05)
    out={'source_sha256':digest(source),'input_receipt_sha256':digest(report),'robot_sha256':digest(artifact),
         'script_sha256':digest(__file__),'urdf_sha256':digest(urdf),'scene_sha256':digest(scene),
         'fps':fps,'frames':frames.tolist(),'duration_s':float(source_times[-1]),
         'unit_quaternion_max_error':float(np.max(np.abs(norm-1))),
         'max_joint_limit_violation_rad':float(violation.max()),
         'mean_absolute_human_robot_keypoint_error_m':float(world_error.mean()),
         'max_absolute_human_robot_keypoint_error_m':float(world_error.max()),
         'static':static,'dynamic_20hz_two_frames':dynamic,
         'warning':'Seven frames / 0.05 s only, raw 120 Hz output. Human-robot world error ignores body proportion and is not a paper metric. z=0 unverified; MuJoCo filters, gravity and step differ from Isaac Lab; no object, learned balance or temporal control. Two-frame 20 Hz interpolation lacks validated anti-aliasing and is used only for a 0.05 s smoke test, not training data or meaningful long-horizon dynamic acceptance.'}
    with Path(__file__).with_name('audit_report.json').open('x',encoding='utf8') as f:
        json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()

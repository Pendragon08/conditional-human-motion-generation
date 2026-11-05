"""Audit an exploratory shared-offset world alignment against the fixed S4 baseline."""
import hashlib,json,re,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OUT=Path(__file__).resolve().parent
BASE=ROOT/'pyroki_s4_offset_diagnostic_20261030'
sys.path.insert(0,str(ROOT/'foot_contact_fit_20261030'))
from g1_fk import URDFFK
from foot_geometry_audit import foot_spheres,sphere_centers
from waist_lock_ik import _joint_limits
LINKS=('pelvis_contour_link','left_hip_pitch_link','right_hip_pitch_link','left_knee_link','right_knee_link',
       'left_ankle_roll_link','right_ankle_roll_link','left_shoulder_roll_link','right_shoulder_roll_link',
       'left_elbow_link','right_elbow_link','left_wrist_yaw_link','right_wrist_yaw_link')
HUMAN=(0,1,2,4,5,7,8,16,17,18,19,20,21)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def replace_once(s,a,b):
    if s.count(a)!=1:raise ValueError('unexpected code pattern')
    return s.replace(a,b)

def main():
    bm=json.loads((BASE/'manifest.json').read_text());vm=json.loads((OUT/'manifest.json').read_text())
    old=BASE/'retarget_helpers/_utils.py';new=OUT/'retarget_helpers/_utils.py'
    if sha(old)!=bm['instrumented_helper_sha256'] or sha(new)!=vm['instrumented_helper_sha256'] or sha(old)!=vm['base_instrumented_helper_sha256']:
        raise ValueError('helper hash mismatch')
    text=old.read_text()
    text=replace_once(text,'        var_robot_cfg: jaxls.Var[jnp.ndarray],\n        keypoints: jnp.ndarray,\n    ) -> jax.Array:\n        """Soft cost to align the human keypoints to the robot, in the world frame."""',
        '        var_robot_cfg: jaxls.Var[jnp.ndarray],\n        var_offset: OffsetVar,\n        keypoints: jnp.ndarray,\n    ) -> jax.Array:\n        """Soft cost to align the final offset-adjusted robot in the world frame."""')
    text=replace_once(text,'        link_pos = T_world_link.translation()[g1_joint_retarget_indices]\n        keypoint_pos = keypoints[smpl_joint_retarget_indices]',
        '        link_pos = T_world_link.translation()[g1_joint_retarget_indices] + var_values[var_offset]\n        keypoint_pos = keypoints[smpl_joint_retarget_indices]')
    text=replace_once(text,'        pc_alignment_cost(\n            var_Ts_world_root,\n            var_joints,\n            target_keypoints,',
        '        pc_alignment_cost(\n            var_Ts_world_root,\n            var_joints,\n            var_offset,\n            target_keypoints,')
    if text!=new.read_text():raise ValueError('unexpected helper change')
    reports=[json.loads((p/'report.json').read_text()) for p in (BASE,OUT)]
    source=ROOT/'grab_s4_drink_human_source_20261030/human_source.npz';baseline=ROOT/'pyroki_grab_s4_drink_windowed_20261030/window_00.npz'
    if any(sha(source)!=x['source_sha256'] or sha(baseline)!=x['baseline_sha256'] for x in reports):raise ValueError('source mismatch')
    urdf=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
    variant=json.loads((ROOT/'pyroki_grab_temporal_20261030/report.json').read_text())
    if sha(urdf)!=variant['urdf_sha256']:raise ValueError('urdf changed')
    with np.load(baseline,allow_pickle=False) as d:names=d['joint_names'].tolist();ref_frames=d['frames']
    with np.load(source,allow_pickle=False) as d:target=d['human_joints'][2:9][:,HUMAN,:];source_frames=d['input_frame_indices'][2:9]
    if not np.array_equal(ref_frames,source_frames):raise ValueError('source frames differ')
    fk=URDFFK(urdf);offs,radii=foot_spheres(urdf);lo,hi=_joint_limits(urdf,names)
    outputs=[]
    for folder,doc in zip((BASE,OUT),reports):
        artifact=folder/'diagnostic.npz'
        if sha(artifact)!=doc['artifact_sha256'] or sha(folder/'pyroki_offset_diagnostic.py')!=doc['driver_sha256']:
            raise ValueError('artifact or driver changed')
        with np.load(artifact,allow_pickle=False) as d:
            frames=d['frames'];post=d['final_root'];pre=d['raw_root'];q=d['joints'];offset=d['offset']
        if not np.array_equal(frames,ref_frames) or not np.allclose(post[:,4:],pre[:,4:]+offset,atol=1e-7,rtol=0):raise ValueError('timeline or offset mismatch')
        if not np.isfinite(post).all() or not np.isfinite(q).all():raise ValueError('nonfinite pose')
        violation=np.maximum(np.maximum(lo.numpy()-q,q-hi.numpy()),0.)
        quaterr=float(np.max(np.abs(np.linalg.norm(post[:,:4],axis=-1)-1)))
        xyz=fk.forward(post[:,4:],post[:,[1,2,3,0]],q,names,links=LINKS)
        err=float(np.linalg.norm(xyz-target,axis=-1).mean())
        if abs(err-doc['unscaled_world_keypoint_distance_after_offset_mean_m'])>1e-8:raise ValueError('FK mean mismatch')
        centers=sphere_centers(fk,post[:,4:],post[:,[1,2,3,0]],q,names,offs)
        clearance=centers[...,2]-radii[None,:,:]
        velocity=np.linalg.norm(np.diff(centers,axis=0)[...,:2],axis=-1)*120
        log=(folder/'solver.log').read_text();term=re.findall(r'Terminated @ iteration #(\d+):[^\n]*criteria=\[([01 ]+)\]',log)
        if len(term)!=1:raise ValueError('solver termination missing')
        outputs.append({'folder':folder.name,'artifact_sha256':sha(artifact),'source_report_sha256':sha(folder/'report.json'),
                        'solver_log_sha256':sha(folder/'solver.log'),'flags':[int(x) for x in term[0][1].split()],
                        'iterations':int(term[0][0]),'max_joint_limit_violation_rad':float(violation.max()),'quaternion_norm_error':quaterr,
                        'unscaled_mean_world_keypoint_error_m':err,'min_foot_sphere_clearance_assumed_z0_m':float(clearance.min()),
                        'max_foot_sphere_horizontal_speed_mps':float(velocity.max()),'root':post,'joints':q})
    a,b=outputs
    root_diff=float(np.max(np.linalg.norm(a['root'][:,4:]-b['root'][:,4:],axis=-1)))
    root_ang=float((Rotation.from_quat(a['root'][:,[1,2,3,0]]).inv()*Rotation.from_quat(b['root'][:,[1,2,3,0]])).magnitude().max())
    joint_diff=float(np.max(np.abs(a['joints']-b['joints'])))
    for d in outputs:del d['root'];del d['joints']
    out={'script_sha256':sha(__file__),'source_sha256':sha(source),'urdf_sha256':sha(urdf),
         'helper_base_sha256':sha(old),'helper_variant_sha256':sha(new),'only_declared_world_residual_change':True,
         'frames':ref_frames.tolist(),'comparisons':outputs,'max_root_position_difference_m':root_diff,
         'max_root_rotation_difference_rad':root_ang,'max_joint_difference_rad':joint_diff,
         'report_correction':'Variant changes global alignment residual as well as returning diagnostic variables; its source report inherited an incomplete description.',
         'limitations':'One overlapping 7-frame tuning window, approximate v1 losses and assumed z=0. Foot sphere clearance, low-speed mask, flags and point distances are not measured contact, full dynamics, paper v2 implementation or training acceptance.'}
    with (OUT/'independent_comparison.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()

"""Raw GRAB SMPL-X -> G1 29-DoF -> 20 Hz -> 27-DoF exploratory pilot.

GMR is not DreamControl-v2's unpublished PyRoki/JAX retargeter. No trajectory
from this pilot is accepted or filtered for training. Keep the source intact.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import torch
import smplx
import trimesh
import mujoco

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
GMR = Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/GMR')
PAIR = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021')
MODEL = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
sys.path.insert(0, str(ROOT / 'foot_contact_fit_20261030'))
sys.path.insert(0, str(GMR))
from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.utils.smpl import get_smplx_data
from general_motion_retargeting.params import IK_CONFIG_DICT
from grab_restricted_reader import read_grab_field
from motion_bridge import SOURCE_NAMES
from omniretarget_bridge import convert_layout
from resample_antialias_omniretarget import resample_antialias
from waist_lock_root_ik import fit_root_and_waist


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    archive = ROOT / 'data/GRAB/grab__s2.zip'
    expected = 'd77eb7d13cfb3e3d0b81f08ccf9d8d79320fa6541352e6e65173c56d4b01b9b6'
    if sha(archive) != expected:
        raise ValueError('archive hash mismatch')
    member = 's2/airplane_fly_1.npz'
    # A contiguous native-rate span; retain timing and output as a pilot only.
    indices = np.arange(540, 660, dtype=np.int64)
    with zipfile.ZipFile(archive) as z:
        body = read_grab_field(z, member, 'body')
        with np.load(io.BytesIO(z.read(member)), allow_pickle=False) as x:
            fps = float(x['framerate'].item()); nframes = int(x['n_frames'].item())
            gender = str(x['gender'].item()); pca = int(x['n_comps'].item())
    if fps != 120 or indices[-1] >= nframes or gender != 'male' or pca != 24:
        raise ValueError('unexpected original frame metadata')
    if body['vtemp'] != 'tools/subject_meshes/male/s2.ply':
        raise ValueError('unexpected template')
    template = ROOT / 'grab_mesh_stage_20261030' / body['vtemp']
    mesh = trimesh.load(str(template), process=False)
    vertices = np.asarray(mesh.vertices)
    if vertices.shape != (10475, 3) or not np.isfinite(vertices).all():
        raise ValueError('invalid template')
    names = ('global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose',
             'leye_pose','reye_pose','expression','transl')
    tensors = {key: torch.as_tensor(np.asarray(body['params'][key])[indices], dtype=torch.float32) for key in names}
    with contextlib.redirect_stdout(sys.stderr):
        model = smplx.create(str(MODEL), model_type='smplx', gender=gender, num_betas=10,
                             num_pca_comps=pca, batch_size=len(indices), use_pca=True, ext='npz')
        if not np.array_equal(np.asarray(mesh.faces), np.asarray(model.faces)):
            raise ValueError('subject template face indices differ')
        model.v_template.copy_(torch.as_tensor(vertices, dtype=model.v_template.dtype))
        with torch.no_grad():
            human = model(**tensors, return_full_pose=True, return_verts=False)
        # This assumed height is an explicit approximation, not measured subject height.
        gmr = GeneralMotionRetargeting(src_human='smplx', tgt_robot='unitree_g1',
                                      actual_human_height=1.66, verbose=False)
    addresses = []
    for name in SOURCE_NAMES:
        jid = mujoco.mj_name2id(gmr.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError('GMR robot missing joint ' + name)
        addresses.append(int(gmr.model.jnt_qposadr[jid]))
    output = []
    for i in range(len(indices)):
        human_frame = get_smplx_data({}, model, human, i)
        q = gmr.retarget(human_frame)
        output.append(np.r_[q[3:7], q[:3], q[addresses]])
        if (i + 1) % 30 == 0:
            print('retargeted', i + 1, '/', len(indices), flush=True)
    q120 = np.asarray(output)
    if q120.shape != (len(indices), 36) or not np.isfinite(q120).all():
        raise ValueError('invalid GMR output')
    # Paper specifies retargeting before 20 Hz sampling; FIR is an approximation.
    q20 = resample_antialias(q120, 120., 20.)
    robot, _ = convert_layout(q20)
    project = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
    assets = project / 'Training/HumanoidVerse/humanoidverse/data/robots/g1'
    u29, u27 = assets/'g1_29dof.urdf', assets/'g1_27dof.urdf'
    fit = fit_root_and_waist(robot, u29, u27, steps=120, learning_rate=.015,
                             foot_position_weight=2., contact_velocity_weight=10.)
    artifact = Path(__file__).with_name('pilot_arrays.npz')
    with artifact.open('xb') as f:
        np.savez_compressed(f, input_frame_indices=indices, input_fps=np.array(fps),
                            human_joints=human.joints[:, :22].detach().cpu().numpy(),
                            g1_qpos_120_wxyz_xyz_29=q120, g1_qpos_20_wxyz_xyz_29=q20,
                            g1_joint_names_29=np.asarray(SOURCE_NAMES),
                            g1_27_root_xyz=fit['root_pos'], g1_27_root_quat_xyzw=fit['root_quat_xyzw'],
                            g1_27_joint_names=np.asarray(fit['joint_names']), g1_27_joint_pos=fit['joint_pos'])
    report = {'source_member': member, 'archive_sha256': expected, 'subject_template_sha256': sha(template),
              'model_sha256': sha(MODEL/'smplx/SMPLX_MALE.npz'), 'gmr_ik_config_sha256': sha(IK_CONFIG_DICT['smplx']['unitree_g1']),
              'gmr_robot_xml_sha256': sha(gmr.xml_file), 'urdf29_sha256': sha(u29), 'urdf27_sha256': sha(u27),
              'script_sha256': sha(__file__), 'artifact_sha256': sha(artifact),
              'input_first_frame': int(indices[0]), 'input_last_frame': int(indices[-1]), 'input_fps': fps,
              'output_fps': 20, 'output_frames': len(q20), 'assumed_human_height_m': 1.66,
              'fit_29_to_27_max_keypoint_error_m': fit['final_max_keypoint_error_m'],
              'fit_29_to_27_mean_keypoint_error_m': fit['final_mean_keypoint_error_m'],
              'warning': 'Pilot only: GMR/MuJoCo is not the authors PyRoki/JAX optimizer, GMR goals, scaling and weights differ, and assumed height is not measured. Locked-head SMPL-X model version may differ. GMR is per-frame with temporal initialization and no validated paper foot-slip objective. FIR quaternion components is approximate. No contact, dynamics, complete text, physical validation, dataset filtering or training acceptance.'}
    with Path(__file__).with_name('report.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print('completed pilot', flush=True)


if __name__ == '__main__':
    main()

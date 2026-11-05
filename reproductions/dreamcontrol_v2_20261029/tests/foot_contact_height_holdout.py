"""Evaluate a fixed near-ground penalty on six earlier held-out box groups.

The six filename-derived groups and ground-penalty baseline were selected
before this height term. Recording independence, z=0 floor and true contact
are unproven. Static MuJoCo queries do not test task objects or dynamics.
"""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
import mujoco
from foot_geometry_audit import foot_spheres,sphere_centers,compare
from g1_fk import URDFFK
from motion_bridge import SOURCE_NAMES
from mujoco_kinematic_contacts import scan
from mujoco_dynamic_contact_witness import digest
from omniretarget_bridge import convert_layout
from resample_omniretarget import resample
from waist_lock_root_ik import fit_root_and_waist


def main():
    torch.set_num_threads(1)
    root=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
    selection_file=root/'foot_ground_fit_20261030/holdout_report.json'
    baseline_file=root/'antialias_mujoco_20261030/kinematic_report.json'
    selected=json.loads(selection_file.read_text())['samples']
    baseline=json.loads(baseline_file.read_text())['samples']
    if len(selected)!=6 or len(baseline)!=6: raise ValueError('expected six previous clips')
    baseline_by_member={x['member']:x for x in baseline}
    if len(baseline_by_member)!=6: raise ValueError('duplicate prior baseline paths')
    project=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
    assets=project/'Training/HumanoidVerse/humanoidverse/data/robots/g1'
    scene=project/'Sim2Real/resources/robots/g1_description/scene_full.xml'
    u29,u27=assets/'g1_29dof.urdf',assets/'g1_27dof.urdf'
    offs29,r29=foot_spheres(u29); offs27,r27=foot_spheres(u27)
    if not (np.allclose(offs29,offs27,rtol=0,atol=1e-12) and np.allclose(r29,r27,rtol=0,atol=1e-12)):
        raise ValueError('foot collision spheres differ')
    fk29,fk27=URDFFK(u29),URDFFK(u27); model=mujoco.MjModel.from_xml_path(str(scene)); samples=[]
    archive_path=root/'data/OmniRetarget_Dataset/robot-object.zip'
    with zipfile.ZipFile(archive_path) as archive:
        for item in selected:
            prior=baseline_by_member[item['member']]
            old=next((m for m in prior['methods'] if m['ground_weight']==10.),None)
            if old is None: raise ValueError('missing expected baseline')
            with np.load(io.BytesIO(archive.read(item['member'])),allow_pickle=False) as data:
                q=resample(data['qpos'],float(data['fps']))[:200]
            motion,_=convert_layout(q)
            source=sphere_centers(fk29,motion[:,:3],motion[:,3:7],motion[:,7:],SOURCE_NAMES,offs29)
            fit=fit_root_and_waist(motion,u29,u27,steps=120,learning_rate=.015,foot_position_weight=2.,
                                   contact_velocity_weight=10.,ground_penetration_weight=10.,ground_plane_z=0.,
                                   ground_contact_weight=10.,ground_contact_band_m=.005)
            target=sphere_centers(fk27,fit['root_pos'],fit['root_quat_xyzw'],fit['joint_pos'],fit['joint_names'],offs27)
            candidate={'ground_weight':10.,'height_weight':10.,'max_keypoint_error_m':fit['final_max_keypoint_error_m'],
                       'foot_metrics':compare(source,target,r27,fps=20,plane_z=0.,height_tolerance=.02,max_vertical_speed=.1),
                       'sim_contacts':scan(model,fit)}
            samples.append({'subject':item['subject'],'member':item['member'],'baseline':old,'candidate':candidate})
    print(json.dumps({'samples':samples,'steps':120,'contact_band_m':.005,'mujoco_version':mujoco.__version__,
                      'input_hashes':{'selection_sha256':digest(selection_file),'baseline_sha256':digest(baseline_file),
                                      'scene_sha256':digest(scene),'urdf27_sha256':digest(u27),'urdf29_sha256':digest(u29),
                                      'archive_sha256':digest(archive_path)},
                      'warning':'Six box motions, different tuning subjects; filename group exclusion is not proof of recording independence. Flat z=0 and source near-ground contact are assumptions. No object, dynamics, Isaac Lab or physical acceptance.'},allow_nan=False))


if __name__=='__main__': main()

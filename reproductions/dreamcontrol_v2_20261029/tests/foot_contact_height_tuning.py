"""Tune near-plane foot height preservation on three box tuning clips.

Plane and source contact are assumptions. MuJoCo queries are static, without
the box or dynamics. These are not independent samples or physical acceptance.
"""
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
from omniretarget_bridge import convert_layout
from resample_omniretarget import resample
from waist_lock_root_ik import fit_root_and_waist


def main():
    torch.set_num_threads(1)
    root=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
    selection=json.loads((root/'tail_fit_20261030/subject_holdout_report.json').read_text())['samples']
    project=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
    assets=project/'Training/HumanoidVerse/humanoidverse/data/robots/g1'
    scene=project/'Sim2Real/resources/robots/g1_description/scene_full.xml'
    u29,u27=assets/'g1_29dof.urdf',assets/'g1_27dof.urdf'
    offs29,r29=foot_spheres(u29); offs27,r27=foot_spheres(u27)
    if not (np.allclose(offs29,offs27,rtol=0,atol=1e-12) and np.allclose(r29,r27,rtol=0,atol=1e-12)):
        raise ValueError('foot sphere geometry differs')
    fk29,fk27=URDFFK(u29),URDFFK(u27); model=mujoco.MjModel.from_xml_path(str(scene))
    names=[name for name in SOURCE_NAMES if name not in ('waist_roll_joint','waist_pitch_joint')]
    indices=[SOURCE_NAMES.index(name) for name in names]
    samples=[]
    with zipfile.ZipFile(root/'data/OmniRetarget_Dataset/robot-object.zip') as archive:
        for subject in (3,7,12):
            choices=[x for x in selection if x['subject']==subject]
            if len(choices)!=3: raise ValueError('unexpected selection')
            member=choices[0]['member']
            with np.load(io.BytesIO(archive.read(member)),allow_pickle=False) as data:
                q=resample(data['qpos'],float(data['fps']))[:200]
            motion,_=convert_layout(q)
            source=sphere_centers(fk29,motion[:,:3],motion[:,3:7],motion[:,7:],SOURCE_NAMES,offs29)
            nominal={'root_pos':motion[:,:3],'root_quat_xyzw':motion[:,3:7],
                     'joint_pos':motion[:,7:][:,indices],'joint_names':names}
            nominal_feet=sphere_centers(fk27,nominal['root_pos'],nominal['root_quat_xyzw'],nominal['joint_pos'],names,offs27)
            if not np.allclose(source,nominal_feet,rtol=0,atol=1e-7): raise ValueError('nominal source feet mismatch')
            methods=[]
            for penetration,contact in ((0.,0.),(10.,0.),(10.,10.),(10.,100.)):
                fit=fit_root_and_waist(motion,u29,u27,steps=120,learning_rate=.015,
                                       foot_position_weight=2.,contact_velocity_weight=10.,
                                       ground_penetration_weight=penetration,ground_plane_z=0.,
                                       ground_contact_weight=contact,ground_contact_band_m=.005)
                target=sphere_centers(fk27,fit['root_pos'],fit['root_quat_xyzw'],fit['joint_pos'],fit['joint_names'],offs27)
                methods.append({'penetration_weight':penetration,'height_weight':contact,
                                'max_keypoint_error_m':fit['final_max_keypoint_error_m'],
                                'foot_metrics':compare(source,target,r27,fps=20,plane_z=0.,height_tolerance=.02,max_vertical_speed=.1),
                                'sim_contacts':scan(model,fit)})
            samples.append({'subject':subject,'member':member,'source_contact_baseline':scan(model,nominal),'methods':methods})
    print(json.dumps({'samples':samples,'contact_band_m':.005,'steps':120,'mujoco_version':mujoco.__version__,
                      'warning':'Three tuning clips only. Source near-ground labels and z=0 assumed; static MuJoCo, no box or dynamics. No independent or physical acceptance.'},allow_nan=False))


if __name__=='__main__': main()

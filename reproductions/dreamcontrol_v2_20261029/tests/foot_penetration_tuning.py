"""Tune an assumed flat-ground penalty on three prior box samples only.

All configurations use equal iterations. This is source-selected tuning, not
independent validation, true contact detection or physics simulation.
"""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from foot_geometry_audit import foot_spheres,sphere_centers,compare
from g1_fk import URDFFK
from motion_bridge import SOURCE_NAMES
from omniretarget_bridge import convert_layout
from resample_omniretarget import resample
from waist_lock_root_ik import fit_root_and_waist


def main():
    torch.set_num_threads(1)
    root=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
    selected=json.loads((root/'tail_fit_20261030/subject_holdout_report.json').read_text())['samples']
    assets=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')
    u29,u27=assets/'g1_29dof.urdf',assets/'g1_27dof.urdf'
    p29,r29=foot_spheres(u29); p27,r27=foot_spheres(u27)
    if not (np.allclose(p29,p27,rtol=0,atol=1e-12) and np.allclose(r29,r27,rtol=0,atol=1e-12)):
        raise ValueError('foot spheres differ')
    fk29,fk27=URDFFK(u29),URDFFK(u27); samples=[]
    with zipfile.ZipFile(root/'data/OmniRetarget_Dataset/robot-object.zip') as archive:
        for subject in (3,7,12):
            subset=[x for x in selected if x['subject']==subject]
            if len(subset)!=3: raise ValueError('unexpected selection')
            member=subset[0]['member']
            with np.load(io.BytesIO(archive.read(member)),allow_pickle=False) as data:
                q=resample(data['qpos'],float(data['fps']))[:200]
            motion,_=convert_layout(q)
            source=sphere_centers(fk29,motion[:,:3],motion[:,3:7],motion[:,7:],SOURCE_NAMES,p29)
            configs=[]
            for weight in (0.,1.,10.,100.):
                fit=fit_root_and_waist(motion,u29,u27,steps=120,learning_rate=.015,
                                       foot_position_weight=2.,contact_velocity_weight=10.,
                                       ground_penetration_weight=weight,ground_plane_z=0.)
                target=sphere_centers(fk27,fit['root_pos'],fit['root_quat_xyzw'],fit['joint_pos'],fit['joint_names'],p27)
                metrics=compare(source,target,r27,fps=20,plane_z=0.,height_tolerance=.02,max_vertical_speed=.1)
                configs.append({'weight':weight,'mean_keypoint_error_m':fit['final_mean_keypoint_error_m'],
                                'max_keypoint_error_m':fit['final_max_keypoint_error_m'],'metrics':metrics})
            samples.append({'subject':subject,'member':member,'frames':len(motion),'configs':configs})
    print(json.dumps({'samples':samples,'steps':120,'learning_rate':.015,
                      'warning':'Tuning only: assumed z=0 flat plane, source-derived stance, linear resampling; no independent physical or dynamics verification.'},allow_nan=False))


if __name__=='__main__': main()

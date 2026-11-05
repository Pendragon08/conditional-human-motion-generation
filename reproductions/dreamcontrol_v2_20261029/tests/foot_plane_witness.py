"""Exploratory foot sphere/flat-plane diagnostics on three box motions.

Source terrain is not available in this check. z=0 is an explicit assumption,
not a verified ground plane, and ankle/foot contact is inferred kinematically.
The outputs do not establish collision, contact or dynamic validity.
"""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from foot_geometry_audit import foot_spheres, sphere_centers, compare
from g1_fk import URDFFK
from motion_bridge import SOURCE_NAMES
from omniretarget_bridge import convert_layout
from resample_omniretarget import resample
from waist_lock_root_ik import fit_root_and_waist


def main():
    torch.set_num_threads(1)
    root=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
    report=json.loads((root/'tail_fit_20261030/subject_holdout_report.json').read_text())
    assets=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')
    u29,u27=assets/'g1_29dof.urdf',assets/'g1_27dof.urdf'
    p29,r29=foot_spheres(u29); p27,r27=foot_spheres(u27)
    if not (np.allclose(p29,p27,rtol=0,atol=1e-12) and np.allclose(r29,r27,rtol=0,atol=1e-12)):
        raise ValueError('source and target foot collision spheres do not match')
    fk29,fk27=URDFFK(u29),URDFFK(u27); samples=[]
    with zipfile.ZipFile(root/'data/OmniRetarget_Dataset/robot-object.zip') as archive:
        for subject in (3,7,12):
            items=[x for x in report['samples'] if x['subject']==subject]
            if len(items)!=3: raise ValueError('unexpected prior sample selection')
            member=items[0]['member']
            with np.load(io.BytesIO(archive.read(member)),allow_pickle=False) as data:
                q=resample(data['qpos'],float(data['fps']))[:200]
            motion,_=convert_layout(q)
            source=sphere_centers(fk29,motion[:,:3],motion[:,3:7],motion[:,7:],SOURCE_NAMES,p29)
            configs=[]
            for method in ('previous','best_experimental'):
                if method=='previous':
                    fit=fit_root_and_waist(motion,u29,u27,steps=120,learning_rate=.015,tail_weight=0.,foot_position_weight=2.,contact_velocity_weight=10.)
                else:
                    fit=fit_root_and_waist(motion,u29,u27,steps=720,learning_rate=.005,tail_weight=1.,foot_position_weight=2.,contact_velocity_weight=100.,return_best=True)
                target=sphere_centers(fk27,fit['root_pos'],fit['root_quat_xyzw'],fit['joint_pos'],fit['joint_names'],p27)
                metrics=compare(source,target,r27,fps=20,plane_z=0.,height_tolerance=.02,max_vertical_speed=.1)
                configs.append({'method':method,'metrics':metrics})
            samples.append({'subject':subject,'member':member,'frames':len(motion),'configs':configs})
    print(json.dumps({'samples':samples,'sphere_count_per_foot':len(r27[0]),
                      'warning':'Exploratory flat z=0 assumption, source-derived stance proxy, linear resampling; no true terrain/contact or dynamics. Three previously used box clips; configurations have different iteration budgets.'},allow_nan=False))


if __name__=='__main__': main()

"""Tune an exploratory tail-error term on the three worst prior samples.

Selection is intentionally adversarial, not an unbiased benchmark. All output
is diagnostic and should be confirmed on independent held-out motions.
"""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch

from g1_fk import URDFFK
from motion_bridge import SOURCE_NAMES
from omniretarget_bridge import convert_layout
from resample_omniretarget import resample
from waist_lock_ik import FIT_LINKS
from waist_lock_root_ik import fit_root_and_waist


def main():
    torch.set_num_threads(1)
    root=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
    folders=[root/'fit_quality_audit_50_20261030',root/'fit_quality_audit_211_20261030']
    rows=[json.loads(p.read_text()) for d in folders for p in d.glob('*.json') if p.name not in ('manifest.json','COMPLETE.json')]
    rows=[r for r in rows if r['status']=='ok']
    rows.sort(key=lambda r:r['methods']['foot_2_contact_10']['max_keypoint_error_m'],reverse=True)
    chosen=rows[:3]
    repo=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
    assets=repo/'third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1'
    u29,u27=assets/'g1_29dof.urdf',assets/'g1_27dof.urdf'; fk29,fk27=URDFFK(u29),URDFFK(u27)
    samples=[]
    for row in chosen:
        with zipfile.ZipFile(root/'data/OmniRetarget_Dataset'/row['archive']) as archive:
            with np.load(io.BytesIO(archive.read(row['member'])),allow_pickle=False) as data:
                q=resample(data['qpos'],float(data['fps']))
        motion,_=convert_layout(q)
        truth=fk29.forward(motion[:,:3],motion[:,3:7],motion[:,7:],SOURCE_NAMES,links=FIT_LINKS)
        velocity=np.diff(truth[:,4:6],axis=0)*20
        contact=np.linalg.norm(velocity,axis=-1)<.03
        results={}
        for weight in [0., .25, .5, 1., 2.]:
            fit=fit_root_and_waist(motion,u29,u27,steps=120,foot_position_weight=2.,contact_velocity_weight=10.,tail_weight=weight)
            pred=fk27.forward(fit['root_pos'],fit['root_quat_xyzw'],fit['joint_pos'],fit['joint_names'],links=FIT_LINKS)
            err=np.linalg.norm(pred-truth,axis=-1)
            velerr=np.linalg.norm(np.diff(pred[:,4:6],axis=0)*20-velocity,axis=-1)
            frame,joint=np.unravel_index(np.argmax(err),err.shape)
            results[str(weight)]={'mean_error_m':float(err.mean()),'max_error_m':float(err.max()),
                                  'worst_link':FIT_LINKS[joint],'worst_frame':int(frame),
                                  'mean_foot_error_m':float(err[:,4:6].mean()),
                                  'contact_velocity_error_mps':float(velerr[contact].mean()) if contact.any() else None}
        samples.append({'archive':row['archive'],'member':row['member'],'frames':len(motion),'weights':results})
    print(json.dumps({'samples':samples,'tail_fraction':.05,'steps':120,
                      'warning':'Adversarial tuning clips only; no held-out, physical or paper-fidelity validation.'}))


if __name__=='__main__': main()

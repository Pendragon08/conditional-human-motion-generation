"""Exploratory cross-subject flat-plane penalty check on unseen filename groups.

Six original paths from subjects 8/10/16 are selected by filename hash, two
per subject, excluding filename-derived groups previously used in both the 261
audit and worst-tail tuning. Source recording independence and the assumed
z=0 flat ground are not established; this is not physics or dataset acceptance.
"""
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
import numpy as np
import torch
from foot_geometry_audit import foot_spheres,sphere_centers,compare
from g1_fk import URDFFK
from motion_bridge import SOURCE_NAMES
from omniretarget_bridge import convert_layout
from resample_omniretarget import resample
from waist_lock_root_ik import fit_root_and_waist


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def group(member):
    match=re.fullmatch(r'robot-object/(sub\d+_largebox_\d+)_(?:original|rot_\d+|trans_\d+)\.npz',member)
    return match.group(1) if match else None


def main():
    torch.set_num_threads(1)
    root=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
    archive_path=root/'data/OmniRetarget_Dataset/robot-object.zip'
    excluded=set()
    for directory in ('fit_quality_audit_50_20261030','fit_quality_audit_211_20261030'):
        samples=json.loads((root/directory/'manifest.json').read_text())['samples']
        excluded.update(g for s in samples if s['archive']==archive_path.name and (g:=group(s['member'])))
    for sample in json.loads((root/'tail_fit_20261030/report.json').read_text())['samples']:
        if sample['archive']==archive_path.name and (g:=group(sample['member'])): excluded.add(g)
    assets=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')
    u29,u27=assets/'g1_29dof.urdf',assets/'g1_27dof.urdf'
    offsets29,radii29=foot_spheres(u29); offsets27,radii27=foot_spheres(u27)
    if not (np.allclose(offsets29,offsets27,rtol=0,atol=1e-12) and np.allclose(radii29,radii27,rtol=0,atol=1e-12)):
        raise ValueError('source and target foot collision spheres differ')
    fk29,fk27=URDFFK(u29),URDFFK(u27); samples=[]
    with zipfile.ZipFile(archive_path) as archive:
        for subject in (8,10,16):
            prefix=f'robot-object/sub{subject}_largebox_'
            pool=[name for name in archive.namelist() if name.startswith(prefix) and name.endswith('_original.npz')
                  and group(name) is not None and group(name) not in excluded]
            pool.sort(key=lambda x:hashlib.sha256(x.encode()).hexdigest())
            if len(pool)<2: raise ValueError(f'insufficient unseen filename groups for subject {subject}')
            for member in pool[:2]:
                with np.load(io.BytesIO(archive.read(member)),allow_pickle=False) as data:
                    q=resample(data['qpos'],float(data['fps']))[:200]
                motion,_=convert_layout(q)
                source=sphere_centers(fk29,motion[:,:3],motion[:,3:7],motion[:,7:],SOURCE_NAMES,offsets29)
                configs=[]
                for weight in (0.,10.):
                    fit=fit_root_and_waist(motion,u29,u27,steps=120,learning_rate=.015,foot_position_weight=2.,
                                           contact_velocity_weight=10.,ground_penetration_weight=weight,ground_plane_z=0.)
                    target=sphere_centers(fk27,fit['root_pos'],fit['root_quat_xyzw'],fit['joint_pos'],fit['joint_names'],offsets27)
                    metrics=compare(source,target,radii27,fps=20,plane_z=0.,height_tolerance=.02,max_vertical_speed=.1)
                    configs.append({'ground_weight':weight,'mean_keypoint_error_m':fit['final_mean_keypoint_error_m'],
                                    'max_keypoint_error_m':fit['final_max_keypoint_error_m'],'metrics':metrics})
                samples.append({'subject':subject,'member':member,'derived_group':group(member),'frames':len(motion),'configs':configs})
    print(json.dumps({'samples':samples,'fps':20,'steps':120,'learning_rate':.015,
                      'archive_sha256':sha256(archive_path),'urdf29_sha256':sha256(u29),'urdf27_sha256':sha256(u27),
                      'warning':'Ground penalty tuning subjects differ and prior audit filename groups excluded; subject and true recording independence not proven. Flat z=0 assumed, linear resampling, source contact proxy only; no dynamics.'},allow_nan=False))


if __name__=='__main__': main()

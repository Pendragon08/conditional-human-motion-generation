"""Inspect no-ground-contact result without assuming source floor is known."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0,str(ROOT/'foot_contact_fit_20261030'))
from g1_fk import URDFFK
from foot_geometry_audit import foot_spheres,sphere_centers


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    run=Path(__file__).resolve().parent; p=run/'raw_pilot.npz'
    rec=json.loads((run/'report.json').read_text())
    if sha(p)!=rec['artifact_sha256']:raise ValueError('pilot hash mismatch')
    with np.load(p,allow_pickle=False) as d:
        root=d['robot_root_wxyz_xyz'].copy();q=d['robot_joint_pos'].copy();names=d['robot_joint_names'].tolist()
        human=d['human_keypoints'].copy();fps=float(d['fps'])
    urdf=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
    offsets,radii=foot_spheres(urdf)
    centers=sphere_centers(URDFFK(urdf),root[:,4:],root[:,[1,2,3,0]],q,names,offsets)
    clearance=centers[...,2]-radii[None,:,:]
    speed=np.linalg.norm(np.diff(centers,axis=0)[...,:2],axis=-1)*fps
    result={'artifact_sha256':sha(p),'urdf_sha256':sha(urdf),'script_sha256':sha(__file__),
            'robot_sphere_min_clearance_to_z0_each_frame_m':clearance.min(axis=(1,2)).tolist(),
            'robot_foot_min_clearance_to_z0_per_side_m':clearance.min(axis=(0,2)).tolist(),
            'robot_foot_horizontal_speed_max_mps':float(speed.max()),
            'human_foot_z_range_m':[float(human[:,[10,11],2].min()),float(human[:,[10,11],2].max())],
            'warning':'Seven frames, source world floor is unverified. z=0 is an exploratory plane; clearance and speed neither infer true contact nor test learned dynamics or training acceptance.'}
    with Path(__file__).with_name('feet_report.json').open('x',encoding='utf8') as f:
        json.dump(result,f,allow_nan=False,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

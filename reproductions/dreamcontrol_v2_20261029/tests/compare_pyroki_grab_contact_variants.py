"""Compare matching seven-frame PyRoki outputs with per-frame foot motion."""
import hashlib,json
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
    dirs=[ROOT/'pyroki_grab_pilot_20261030',ROOT/'pyroki_grab_linear_contact_20261030']
    reports=[json.loads((x/'report.json').read_text()) for x in dirs]
    for key in ('source_sha256','urdf_sha256','source_frames','fps','weights','human_indices','robot_indices'):
        if reports[0][key]!=reports[1][key]:raise ValueError('configuration mismatch: '+key)
    arrays=[]
    for folder,report in zip(dirs,reports):
        p=folder/'raw_pilot.npz'
        if sha(p)!=report['artifact_sha256']:raise ValueError('artifact hash mismatch')
        with np.load(p,allow_pickle=False) as x: arrays.append({k:x[k].copy() for k in x.files})
    for key in ('frames','fps','human_keypoints','robot_joint_names'):
        if not np.array_equal(arrays[0][key],arrays[1][key]):raise ValueError('input mismatch: '+key)
    urdf=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
    if sha(urdf)!=reports[0]['urdf_sha256']:raise ValueError('URDF mismatch')
    fk=URDFFK(urdf);offsets,radii=foot_spheres(urdf);out=[]
    for label,x in zip(('squared','linear'),arrays):
        root=x['robot_root_wxyz_xyz']; q=x['robot_joint_pos'];fps=float(x['fps'])
        spheres=sphere_centers(fk,root[:,4:],root[:,[1,2,3,0]],q,x['robot_joint_names'].tolist(),offsets)
        clear=(spheres[...,2]-radii[None,:,:]).min(axis=(1,2))
        horizontal=np.linalg.norm(np.diff(spheres,axis=0)[...,:2],axis=-1)*fps
        per_interval=horizontal.max(axis=(1,2))
        out.append({'variant':label,'report_sha256':sha(dirs[len(out)]/'report.json'),
                    'artifact_sha256':sha(dirs[len(out)]/'raw_pilot.npz'),
                    'foot_clearance_per_frame_m':clear.tolist(),
                    'foot_max_horizontal_speed_per_interval_mps':per_interval.tolist(),
                    'robot_root_translation_speed_per_interval_mps':(np.linalg.norm(np.diff(root[:,4:],axis=0),axis=1)*fps).tolist(),
                    'max_joint_speed_per_interval_radps':(np.abs(np.diff(q,axis=0))*fps).max(axis=1).tolist()})
    result={'source_sha256':reports[0]['source_sha256'],'urdf_sha256':sha(urdf),'script_sha256':sha(__file__),
            'frames':arrays[0]['frames'].tolist(),'fps':float(arrays[0]['fps']),'samples':out,
            'warning':'Seven consecutive frames of one subject and action; 120Hz finite differences are sensitive to noise and solver boundary effects. z=0 and low-speed contacts are assumptions; PyRoki uses v1 objectives and approximate weights; not true contact, dynamics or training acceptance.'}
    with Path(__file__).with_name('comparison_report.json').open('x',encoding='utf8') as f:
        json.dump(result,f,allow_nan=False,indent=2)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()

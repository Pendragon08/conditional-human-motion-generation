"""Compare diagnostic overlap selection and blending on S2/S3 PyRoki runs."""
import hashlib,json
from pathlib import Path
import sys
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0,str(ROOT/'foot_contact_fit_20261030'))
sys.path.insert(1,str(ROOT/'antialias_mujoco_20261030'))
from blend_robot_windows import blend_windows
from resample_g1_27 import resample_27
from g1_fk import URDFFK
from foot_geometry_audit import foot_spheres,sphere_centers
from waist_lock_ik import _joint_limits
from mujoco_kinematic_contacts import scan
from mujoco_dynamic_contact_witness import simulate

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    torch.set_num_threads(1)
    project=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
    urdf=project/'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
    scene=project/'Sim2Real/resources/robots/g1_description/scene_full.xml'
    gains=project/'Sim2Real/deploy/deploy_mujoco/configs/g1_full_body.yaml'
    fk=URDFFK(urdf);offs,radii=foot_spheres(urdf);model=mujoco.MjModel.from_xml_path(str(scene))
    summaries=[]
    for label,folder in [('s2',ROOT/'pyroki_grab_windowed_20261030'),('s3',ROOT/'pyroki_grab_s3_windowed_20261030')]:
        report=json.loads((folder/'report.json').read_text()); audit=json.loads((folder/'audit_report.json').read_text())
        if sha(folder/'report.json')!=audit['report_sha256'] or sha(folder/'windows.jsonl')!=report['rows_sha256']:
            raise ValueError('audited input changed')
        if sha(urdf)!=report['urdf_sha256']:raise ValueError('URDF mismatch')
        rows=[json.loads(x) for x in (folder/'windows.jsonl').read_text().splitlines()]
        if len(rows)!=29 or audit['windows_verified']!=29:raise ValueError('incomplete input')
        windows=[]
        for row in rows:
            p=folder/row['artifact']
            if sha(p)!=row['artifact_sha256']:raise ValueError('window hash mismatch')
            with np.load(p,allow_pickle=False) as d:
                windows.append({k:d[k].copy() for k in ('frames','root_wxyz_xyz','joint_pos','joint_names')})
        result=blend_windows(windows); frames=result['frames'];names=list(result['joint_names'])
        if len(frames)!=118 or np.any(np.diff(frames)!=1):raise ValueError('unexpected coverage')
        lo,hi=_joint_limits(urdf,names);lo=lo.numpy();hi=hi.numpy();methods=[]
        for method in ('first','blended'):
            q=result[method]; q20=resample_27(q); sampled_frames=frames[0]+np.arange(len(q20))*6
            if len(q20)!=20 or sampled_frames[-1]>frames[-1]:raise ValueError('output timeline mismatch')
            fit={'root_pos':q20[:,4:7], 'root_quat_xyzw':q20[:,[1,2,3,0]],
                 'joint_pos':q20[:,7:], 'joint_names':names}
            centers=sphere_centers(fk,q[:,4:7],q[:,[1,2,3,0]],q[:,7:],names,offs)
            horizontal=np.linalg.norm(np.diff(centers,axis=0)[...,:2],axis=-1)*120
            root_v=np.linalg.norm(np.diff(q[:,4:7],axis=0),axis=-1)*120
            joint_v=np.abs(np.diff(q[:,7:],axis=0))*120
            angle=(Rotation.from_quat(q[:-1,[1,2,3,0]]).inv()*Rotation.from_quat(q[1:,[1,2,3,0]])).magnitude()*120
            viol=np.maximum(np.maximum(lo-q20[:,7:],q20[:,7:]-hi),0.)
            out=Path(__file__).with_name(f'{label}_{method}.npz')
            with out.open('xb') as f:np.savez_compressed(f,frames120=frames,qpos120=q,frames20=sampled_frames,qpos20=q20,joint_names=np.asarray(names))
            dynamic=simulate(scene,gains,urdf,fit,seconds=.9)
            methods.append({'method':method,'artifact_sha256':sha(out),'output_frames':len(q20),
                'max_120hz_foot_horizontal_speed_mps':float(horizontal.max()),
                'p95_120hz_foot_horizontal_speed_mps':float(np.percentile(horizontal,95)),
                'max_120hz_root_speed_mps':float(root_v.max()),'max_120hz_joint_speed_radps':float(joint_v.max()),
                'max_120hz_root_angular_speed_radps':float(angle.max()),
                'max_20hz_joint_limit_violation_rad':float(viol.max()),
                'static':scan(model,fit),'dynamic':dynamic})
            print('done',label,method,flush=True)
        summaries.append({'subject':label,'source_report_sha256':sha(folder/'report.json'),'methods':methods})
    report={'samples':summaries,'blend_code_sha256':sha(Path(__file__).with_name('blend_robot_windows.py')),
            'resample_code_sha256':sha(Path(__file__).with_name('resample_g1_27.py')),'script_sha256':sha(__file__),
            'urdf_sha256':sha(urdf),'scene_sha256':sha(scene),'gains_sha256':sha(gains),
            'warning':'Two subject/actions, fixed-window blending and component-FIR are experimental, not PyRoki global solve or author pipeline. Explicit frame continuity does not establish physically correct motion. z=0 ground unverified, old MuJoCo uses gravity -7.5, dt .002, hand geometries and no object/learned balance. No physical or training acceptance.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print('complete',flush=True)

if __name__=='__main__':main()

"""Blend audited source-rate GRAB robot windows and export 20 Hz previews.

FIR filtering of quaternion components is approximate; object poses are sampled
without antialiasing. Contact, text, source clocks and physics remain unaccepted.
"""
import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from blend_robot_windows import blend_windows
from feature263 import canonicalize_z_up, encode
from g1_fk import URDFFK
from resample_g1_27 import resample_27
import blend_robot_windows, feature263, g1_fk, resample_g1_27

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
WORK=ROOT/'grab_full_robot_work_20261102'
HUMAN=ROOT/'grab_full_human_source_batch_20261102'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    audit_path=WORK/'robot_windows_audit_20261102.json';audit_hash=sha(audit_path)
    audit=json.loads(audit_path.read_text());cases={r['id']:r for r in audit['cases']}
    if set(cases)!={'000','001'} or sha(URDF)!=audit['urdf_sha256']:raise ValueError('audit binding mismatch')
    limits={}
    for joint in ET.parse(URDF).getroot().findall('joint'):
        lim=joint.find('limit')
        if joint.get('type')=='revolute' and lim is not None:limits[joint.get('name')]=(float(lim.get('lower')),float(lim.get('upper')))
    modules=[Path(m.__file__).resolve() for m in (blend_robot_windows,feature263,g1_fk,resample_g1_27)]
    module_hashes={str(m):sha(m) for m in modules}
    if shutil.disk_usage(a.out.parent).free<8*1024**3+64*1024**2:raise RuntimeError('disk budget')
    a.out.mkdir(exist_ok=False);results=[]
    for sid in ('000','001'):
        result={'id':sid,'training_ready':False,'caption_accepted':False}
        try:
            if cases[sid]['joint_values_outside_declared_limits']!=0:raise ValueError('source windows exceed joint limits')
            d=WORK/'robot_windows'/sid;source=HUMAN/(sid+'.npz')
            report=json.loads((d/'report.json').read_text())
            if (sha(d/'report.json')!=cases[sid]['solver_report_sha256'] or sha(source)!=cases[sid]['human_sha256']
                or sha(d/'windows.jsonl')!=report['rows_sha256'] or report['source_declared_fps']!=120.):
                raise ValueError('input hash or declared rate mismatch')
            with np.load(source,allow_pickle=False) as f:
                source_frames=f['source_frames'].copy();object_xyz=f['object_transl'].copy();object_rot=f['object_orient_rotvec'].copy()
            rows=[json.loads(line) for line in (d/'windows.jsonl').read_text().splitlines()];windows=[]
            for row in rows:
                target=d/row['artifact']
                if sha(target)!=row['sha256']:raise ValueError('window hash mismatch')
                with np.load(target,allow_pickle=False) as f:
                    windows.append({'frames':f['sample_indices'].copy(),'root_wxyz_xyz':f['root_wxyz_xyz'].copy(),
                        'joint_pos':f['joint_pos'].copy(),'joint_names':f['joint_names'].copy()})
            merged=blend_windows(windows);frames=merged['frames'];np.testing.assert_array_equal(frames,np.arange(1,len(source_frames)))
            full_state=merged['blended'];names=list(merged['joint_names'])
            if set(names)!=set(limits):raise ValueError('actuator names mismatch')
            state=resample_27(full_state,source_fps=120.,target_fps=20.)
            indices=frames[np.arange(len(state))*6]
            lo=np.array([limits[x][0] for x in names]);hi=np.array([limits[x][1] for x in names])
            excess=float(np.maximum(np.maximum(lo-state[:,7:],state[:,7:]-hi),0).max())
            if excess>0:raise ValueError('filtered state exceeds declared joint limits')
            points=URDFFK(URDF).forward(state[:,4:7],state[:,[1,2,3,0]],state[:,7:],names)
            features=encode(points,foot_velocity_squared_threshold=.002)
            with torch.no_grad():decoded=recover_from_ric(torch.from_numpy(features),22).numpy()
            error=float(np.linalg.norm(decoded-canonicalize_z_up(points)[:-1],axis=-1).max())
            if features.shape!=(len(state)-1,263) or not np.isfinite(features).all() or error>1e-4:
                raise ValueError('feature encoding self-check failed')
            output=a.out/(sid+'.npz')
            with output.open('xb') as f:
                np.savez_compressed(f,features=features,root_wxyz_xyz=state[:,:7],joint_pos=state[:,7:],
                    sample_times=indices.astype(np.float64)/120.,source_frames=indices,joint_names=np.asarray(names),
                    points_z_up_world=points,object_transl=object_xyz[indices],object_orient_rotvec=object_rot[indices])
            result.update({'status':'technical_preview','artifact':output.name,'output_sha256':sha(output),
                'human_sha256':sha(source),'solver_report_sha256':sha(d/'report.json'),'window_audit_sha256':audit_hash,
                'states':len(state),'features':len(features),'source_frame_first_last':[int(indices[0]),int(indices[-1])],
                'max_joint_limit_excess_rad':excess,'decode_selfcheck_max_error_m':error})
        except (ValueError,RuntimeError,KeyError,IndexError) as exc:
            result.update({'status':'quarantined','reason':f'{type(exc).__name__}: {exc}'})
        results.append(result)
    if sha(audit_path)!=audit_hash or any(sha(p)!=module_hashes[str(p)] for p in modules):raise RuntimeError('input changed')
    final={'cases':results,'window_audit_sha256':audit_hash,'module_sha256':module_hashes,
           'script_sha256':sha(Path(__file__)),'training_ready':False,
           'limitations':['Source-declared FPS is not verified capture time; input first frame is missing.',
             'Weighted window blending and 120-to-20 polyphase FIR on quaternion components are approximations.',
             'Object poses are sampled at corresponding source frames without antialiasing.',
             'Same FK and encoder for self-check; captions, contacts and physical quality not accepted.']}
    with (a.out/'report.json').open('x',encoding='utf8') as f:json.dump(final,f,indent=2,allow_nan=False)
    print(json.dumps({'cases':len(results),'preview':sum(r['status']=='technical_preview' for r in results),
                      'quarantined':sum(r['status']=='quarantined' for r in results),'training_ready':False}))


if __name__=='__main__':main()

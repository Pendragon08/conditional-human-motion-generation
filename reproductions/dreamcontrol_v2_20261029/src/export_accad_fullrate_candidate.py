"""Blend audited 120-Hz robot windows, then filter and encode at nominal 20 Hz.

This is an isolated unaccepted candidate, not the authors' v2 method. Filtering,
contact, source identity, original time and caption semantics remain unverified.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from blend_robot_windows import blend_windows
from resample_g1_27 import resample_27
from feature263 import canonicalize_z_up, encode
from g1_fk import URDFFK

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
RUN=ROOT/'accad_013267_pyroki_fullrate_20261031'
HUMAN=ROOT/'accad_013267_fullrate_20261031'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    result=json.loads((RUN/'report.json').read_text());check=json.loads((RUN/'audit_report.json').read_text())
    human=json.loads((HUMAN/'human_report.json').read_text())
    if (sha(RUN/'report.json')!=check['report_sha256'] or sha(HUMAN/'human_120hz.npz')!=check['source_sha256']
        or sha(URDF)!=check['urdf_sha256'] or check['windows_verified']!=88 or check['covered_indices']!=[1,354]):
        raise ValueError('window audit mismatch')
    rows=[json.loads(line) for line in (RUN/'windows.jsonl').read_text().splitlines()]
    if sha(RUN/'windows.jsonl')!=result['rows_sha256']:raise ValueError('window manifest changed')
    windows=[]
    for row in rows:
        path=RUN/row['artifact']
        if sha(path)!=row['artifact_sha256']:raise ValueError('window changed')
        with np.load(path,allow_pickle=False) as f:
            windows.append({'frames':f['source_frames'].copy(),'root_wxyz_xyz':f['root_wxyz_xyz'].copy(),
                            'joint_pos':f['joint_pos'].copy(),'joint_names':f['joint_names'].copy()})
    merged=blend_windows(windows);frames=merged['frames'];
    np.testing.assert_array_equal(frames,np.arange(1,355))
    start=int(np.flatnonzero(frames==6)[0]);use=merged['blended'][start:]
    filtered=resample_27(use,source_fps=120.,target_fps=20.)
    out_frames=frames[start::6]
    if filtered.shape!=(59,34) or len(out_frames)!=59 or out_frames[-1]!=354:raise ValueError('bad output grid')
    root=filtered[:,:7];q=filtered[:,7:];names=list(merged['joint_names'])
    points=URDFFK(URDF).forward(root[:,4:],root[:,[1,2,3,0]],q,names)
    features=encode(points,foot_velocity_squared_threshold=.002)
    with torch.no_grad():decoded=recover_from_ric(torch.from_numpy(features),22).numpy()
    error=float(np.linalg.norm(decoded-canonicalize_z_up(points)[:-1],axis=-1).max())
    if features.shape!=(58,263) or not np.isfinite(features).all() or error>1e-4:
        raise ValueError('candidate encoding failed self-check')
    limits={}
    for joint in ET.parse(URDF).getroot().findall('joint'):
        if joint.get('type')=='continuous':limits[joint.get('name')]=(-np.inf,np.inf)
        elif joint.get('type')=='revolute':
            n=joint.find('limit');limits[joint.get('name')]=(float(n.get('lower')),float(n.get('upper')))
    if set(names)!=set(limits):raise ValueError('joint mismatch')
    low=np.array([limits[n][0] for n in names]);high=np.array([limits[n][1] for n in names])
    original=merged['blended'][:,7:]
    violation=lambda x:float(np.maximum(np.maximum(low-x,x-high),0).max())
    out=HERE/'candidate.npz'
    with out.open('xb') as f:
        np.savez_compressed(f,features=features,points_z_up_world=points,decoded_points=decoded,
            raw_frames=frames,first_solution=merged['first'],blended_120hz=merged['blended'],
            frames_20hz=out_frames,filtered_20hz=filtered,joint_names=np.asarray(names))
    report={'id':'013267','output_sha256':sha(out),'windows_report_sha256':sha(RUN/'report.json'),
            'windows_audit_sha256':sha(RUN/'audit_report.json'),'human_report_sha256':sha(HUMAN/'human_report.json'),
            'source_sha256':human['source_sha256'],'urdf_sha256':sha(URDF),'script_sha256':sha(Path(__file__)),
            'blend_sha256':sha(HERE/'blend_robot_windows.py'),'resample_sha256':sha(HERE/'resample_g1_27.py'),
            'raw_covered_frames':[int(frames[0]),int(frames[-1])], 'filter_input_frames':[6,354],
            'frame_grid_20hz':out_frames.tolist(),'features_shape':list(features.shape),
            'decode_max_error_m':error,'max_joint_limit_violation_before_filter_rad':violation(original),
            'max_joint_limit_violation_after_filter_rad':violation(q),
            'first_vs_blended_joint_max_rad':float(np.max(np.abs(merged['first'][:,7:]-original))),
            'caption_accepted':False,'captions':human['captions'],'training_ready':False,
            'limitations':['Full-rate old v1 PyRoki, heuristic temporal weights and overlap blend are not author method.',
                           'FIR begins at frame 6 and uses line padding; quaternion component filtering is approximate.',
                           'FPS/source identity, model equivalence, caption semantics and physical quality unaccepted.']}
    with (HERE/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

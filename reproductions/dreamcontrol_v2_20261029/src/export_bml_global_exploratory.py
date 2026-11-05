"""Encode an audited BML robot excerpt as an unaccepted 263D candidate."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from feature263 import canonicalize_z_up, encode
from g1_fk import URDFFK

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    label=sys.argv[1]
    if label not in ('S04_Expert','S08_Novice'):raise ValueError('invalid candidate')
    solve=ROOT/'bml_global_candidates_20261031'/label
    report=json.loads((solve/'report.json').read_text())
    audit=json.loads((solve/'audit_report.json').read_text())
    if (sha(solve/'report.json')!=audit['report_sha256'] or
        sha(solve/'trajectory.npz')!=audit['trajectory_sha256'] or
        audit['trajectory_sha256']!=report['artifact_sha256'] or
        sha(URDF)!=report['urdf_sha256'] or audit['frames_verified']!=60):
        raise ValueError('trajectory audit mismatch')
    with np.load(solve/'trajectory.npz',allow_pickle=False) as d:
        t=d['source_times'].astype(np.float64); frames=d['source_frames'].copy()
        root=d['root_wxyz_xyz'].astype(np.float64); q=d['joint_pos'].astype(np.float64)
        names=[str(x) for x in d['joint_names']]
    if (not np.all(np.diff(frames)==6) or not np.allclose(t,frames/120.,rtol=0,atol=1e-12)
        or not np.allclose(np.diff(t),.05,rtol=0,atol=1e-12)):
        raise ValueError('nominal time grid mismatch')
    pos=root[:,4:]; quat=root[:,[1,2,3,0]]
    points=URDFFK(URDF).forward(pos,quat,q,names)
    features=encode(points,foot_velocity_squared_threshold=.002)
    with torch.no_grad(): decoded=recover_from_ric(torch.from_numpy(features),22).numpy()
    err=float(np.linalg.norm(decoded-canonicalize_z_up(points)[:-1],axis=-1).max())
    if features.shape!=(59,263) or err>1e-4:raise ValueError('invalid encoded output')
    directory=Path.cwd()/label;directory.mkdir()
    artifact=directory/'candidate.npz'
    with artifact.open('xb') as f:
        np.savez_compressed(f,features=features,source_frames=frames,source_times=t,
                            points_z_up_world=points,decoded_points=decoded,
                            root_wxyz_xyz=root,joint_pos=q,joint_names=np.asarray(names))
    out={'label':label,'source_trajectory_sha256':sha(solve/'trajectory.npz'),
         'source_report_sha256':sha(solve/'report.json'),'source_audit_sha256':sha(solve/'audit_report.json'),
         'urdf_sha256':sha(URDF),'script_sha256':sha(Path(__file__)), 'feature_sha256':sha(artifact),
         'source_frames_first_last':[int(frames[0]),int(frames[-1])], 'points':len(points),
         'feature_rows':len(features),'fps_assuming_declared_120hz':20,
         'max_joint_limit_violation_rad':audit['max_joint_limit_violation_rad'],
         'decode_max_error_m':err,'foot_threshold_squared':.002,'training_ready':False,
         'limitations':['excerpt text semantics not accepted; source clip annotations remain in six-source view',
                        'source/model identity, true fps, BML-specific terms, contact and quality not accepted',
                        'old v1 solver and 263D encoder are approximations, no official split altered']}
    with (directory/'report.json').open('x',encoding='utf8') as f:
        json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()

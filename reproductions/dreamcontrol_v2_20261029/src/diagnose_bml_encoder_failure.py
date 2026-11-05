"""Recompute and record the failed S04 encoding; never accept it as data."""
import hashlib
import json
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
    solve=ROOT/'bml_global_candidates_20261031/S04_Expert'
    trajectory=solve/'trajectory.npz'; audit=json.loads((solve/'audit_report_v2.json').read_text())
    assert sha(trajectory)==audit['trajectory_sha256']
    with np.load(trajectory,allow_pickle=False) as d:
        q=d['joint_pos'].copy();root=d['root_wxyz_xyz'].copy(); names=[str(x) for x in d['joint_names']]
    points=URDFFK(URDF).forward(root[:,4:],root[:,[1,2,3,0]],q,names)
    features=encode(points,foot_velocity_squared_threshold=.002)
    with torch.no_grad():decoded=recover_from_ric(torch.from_numpy(features),22).numpy()
    normalized=canonicalize_z_up(points)[:-1]
    errors=np.linalg.norm(decoded-normalized,axis=-1)
    j=int(np.abs(np.diff(q,axis=0)).argmax()); frame,joint=np.unravel_index(j,(59,27))
    out={'trajectory_sha256':sha(trajectory),'source_audit_sha256':sha(solve/'audit_report_v2.json'),
         'feature_shape':list(features.shape),'finite':bool(np.isfinite(features).all()),
         'decode_max_error_m':float(errors.max()),'decode_median_error_m':float(np.median(errors)),
         'error_first_last_m':[float(errors[0].max()),float(errors[-1].max())],
         'max_joint_step_rad':float(np.abs(np.diff(q,axis=0)).max()),
         'max_joint_step_start_row':int(frame),'max_joint_step_name':names[joint],
         'training_ready':False,'limitations':'Diagnostic recomputes same FK/encoder; does not identify physical cause or validate source.'}
    with (Path.cwd()/'failure_diagnostic.json').open('x',encoding='utf8') as f:
        json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()

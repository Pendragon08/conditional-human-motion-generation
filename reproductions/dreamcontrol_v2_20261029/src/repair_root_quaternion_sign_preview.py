"""Test a sign-continuous root-quaternion encoding for one quarantined clip.

This is an isolated experiment: it changes derived features, not source data,
and is not a claim about the authors' v2 implementation or physics.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.common.quaternion import qfix,qinv_np,qmul_np
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets
from feature263 import canonicalize_z_up,encode,CHAINS,FACE
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOURCE=ROOT/'omni_gpu_batch24_20261101';HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sid='sub8_largebox_006';rows={r['sample_id']:r for r in map(json.loads,(SOURCE/'records.jsonl').read_text().splitlines())}
    r=rows[sid];state=SOURCE/r['state20'];feat=SOURCE/r['feature']
    if sha(state)!=r['state20_sha256'] or sha(feat)!=r['feature_sha256']:raise ValueError('source changed')
    with np.load(state,allow_pickle=False) as f:points=f['points_z_up']
    with np.load(feat,allow_pickle=False) as f:original=f['features']
    if not np.array_equal(encode(points,foot_velocity_squared_threshold=.002),original):raise ValueError('source encode changed')
    canonical=canonicalize_z_up(points)
    skel=Skeleton(torch.as_tensor(t2m_raw_offsets,dtype=torch.float32),CHAINS,'cpu')
    quat=skel.inverse_kinematics_np(canonical.copy(),FACE,smooth_forward=True)
    raw=quat[:,0].copy();dots=np.sum(raw[1:]*raw[:-1],axis=1)
    raw_vel=qmul_np(raw[1:],qinv_np(raw[:-1]));fixed=qfix(quat.copy())[:,0]
    fixed_vel=qmul_np(fixed[1:],qinv_np(fixed[:-1]))
    expected=np.arcsin(np.clip(raw_vel[:,2],-1.,1.)).astype(np.float32)
    if not np.array_equal(expected,original[:,0]):raise ValueError('could not reproduce original root angular channel')
    candidate=original.copy();candidate[:,0]=np.arcsin(np.clip(fixed_vel[:,2],-1.,1.)).astype(np.float32)
    with torch.no_grad():
        old=recover_from_ric(torch.from_numpy(original),22).numpy()
        new=recover_from_ric(torch.from_numpy(candidate),22).numpy()
    before=float(np.linalg.norm(old-canonical[:-1],axis=-1).max());after=float(np.linalg.norm(new-canonical[:-1],axis=-1).max())
    path=HERE/'corrected_feature.npz'
    with path.open('xb') as f:np.savez_compressed(f,features=candidate)
    result={'sample_id':sid,'source_state_sha256':sha(state),'source_features_sha256':sha(feat),
        'script_sha256':sha(Path(__file__)),'corrected_features_sha256':sha(path),
        'negative_quaternion_dot_intervals':np.flatnonzero(dots<0).tolist(),
        'angular_channel_changed_rows':np.flatnonzero(candidate[:,0]!=original[:,0]).tolist(),
        'other_channels_unchanged':bool(np.array_equal(candidate[:,1:],original[:,1:])),
        'original_decode_max_error_m':before,'corrected_decode_max_error_m':after,
        'training_ready':False,'limitations':['Uses same experimental forward encoder/decoder and body skeleton library.',
        'Sign correction is a project approximation, not verified author v2 method.',
        'No source/license/text/physical quality acceptance and no automatic training inclusion.']}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

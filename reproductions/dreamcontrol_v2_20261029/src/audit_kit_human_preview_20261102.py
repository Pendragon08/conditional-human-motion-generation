"""Recompute three SMPL-X preview frames and verify nominal timing/hashes."""
import argparse, contextlib, hashlib, json, sys
from pathlib import Path
import numpy as np
import smplx
import torch

MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--preview',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    r=json.loads((a.preview/'report.json').read_text());artifact=a.preview/'human_20hz_declared.npz'
    if sha(a.source)!=r['source_sha256'] or sha(artifact)!=r['artifact_sha256']:raise ValueError('hash mismatch')
    with np.load(a.source,allow_pickle=False) as z:
        trans=z['trans'].copy();body=z['pose_body'].copy();orient=z['root_orient'].copy();betas=z['betas'][:16].astype(np.float32);gender=str(z['gender'].item());fps=float(z['mocap_frame_rate'])
    model_file=MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz')
    if sha(model_file)!=r['model_sha256'] or fps!=r['declared_fps']:raise ValueError('model or clock changed')
    with np.load(artifact,allow_pickle=False) as z:points=z['human'].copy();ix=z['source_frames'].copy();times=z['source_times'].copy()
    if not np.array_equal(ix,np.arange(0,len(trans),6)) or not np.array_equal(times,ix/fps):raise ValueError('nominal timeline mismatch')
    with contextlib.redirect_stdout(sys.stderr):model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=16,use_pca=False,batch_size=1,ext='npz').cpu()
    indices=[0,len(ix)//2,len(ix)-1];maxdiff=0.
    with torch.no_grad():
        for j in indices:
            zero=lambda n:torch.zeros((1,n),dtype=torch.float32)
            t=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32)
            v=model(betas=t(betas[None]),global_orient=t(orient[ix[j]][None]),body_pose=t(body[ix[j]][None]),transl=t(trans[ix[j]][None]),left_hand_pose=zero(45),right_hand_pose=zero(45),jaw_pose=zero(3),leye_pose=zero(3),reye_pose=zero(3),expression=zero(model.num_expression_coeffs),return_verts=False).joints[0,:22].numpy()
            maxdiff=max(maxdiff,float(np.max(np.abs(v-points[j]))))
    if maxdiff>1e-5:raise ValueError('forward comparison exceeded tolerance')
    out={'id':r['id'],'artifact_sha256':sha(artifact),'source_sha256':sha(a.source),'report_sha256':sha(a.preview/'report.json'),'checked_output_indices':indices,'max_abs_difference_m':maxdiff,'training_ready':False,'limitations':['Same model and smplx implementation, not independent physical or human ground truth.','Declared FPS is not a verified clock.']}
    with (a.preview/'audit.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()

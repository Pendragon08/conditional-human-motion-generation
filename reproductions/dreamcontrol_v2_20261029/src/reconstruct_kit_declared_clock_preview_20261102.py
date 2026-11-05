"""Isolated KIT SMPL-X keypoint preview at file-declared 120 Hz.

Not an official HumanML reconstruction, timing witness, or training input.
"""
import argparse, contextlib, hashlib, json, os, shutil, sys
from pathlib import Path
import numpy as np
import smplx
import torch

MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--id',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(1)
    manifest_hash=sha(a.manifest);manifest=json.loads(a.manifest.read_text());cases=[c for c in manifest['members'] if c['id']==a.id]
    if len(cases)!=1:raise ValueError('source ID missing or ambiguous')
    case=cases[0];source_hash=sha(a.source)
    if source_hash!=case['sha256']:raise ValueError('source hash mismatch')
    with np.load(a.source,allow_pickle=False) as z:
        fps=float(z['mocap_frame_rate']); gender=str(z['gender'].item()); model_type=str(z['surface_model_type'].item())
        trans=z['trans'].copy();orient=z['root_orient'].copy();body=z['pose_body'].copy();betas=np.asarray(z['betas'],dtype=np.float32).reshape(-1)[:16]
    n=len(trans)
    if fps!=120.0 or model_type!='smplx' or gender not in ('male','female','neutral') or trans.shape!=(n,3) or orient.shape!=(n,3) or body.shape!=(n,63) or betas.shape!=(16,):raise ValueError('unexpected SMPL-X schema')
    if any(not np.isfinite(x).all() for x in (trans,orient,body,betas)):raise ValueError('nonfinite parameters')
    model_path=MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz');model_hash=sha(model_path)
    if shutil.disk_usage(a.out.parent).free < 8*1024**3+64*1024**2:raise RuntimeError('insufficient disk budget')
    a.out.mkdir(exist_ok=False)
    with contextlib.redirect_stdout(sys.stderr):
        model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=16,use_pca=False,batch_size=1,ext='npz').cpu()
    ix=np.arange(0,n,6,dtype=np.int64); batches=[]
    for start in range(0,len(ix),60):
        frames=ix[start:start+60]; count=len(frames)
        tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32)
        zero=lambda width:torch.zeros((count,width),dtype=torch.float32)
        with torch.no_grad():
            pts=model(betas=tensor(np.broadcast_to(betas,(count,16)).copy()),global_orient=tensor(orient[frames]),body_pose=tensor(body[frames]),transl=tensor(trans[frames]),left_hand_pose=zero(45),right_hand_pose=zero(45),jaw_pose=zero(3),leye_pose=zero(3),reye_pose=zero(3),expression=zero(model.num_expression_coeffs),return_verts=False).joints[:,:22].numpy()
        batches.append(pts)
    pts=np.concatenate(batches)
    if pts.shape!=(len(ix),22,3) or not np.isfinite(pts).all():raise ValueError('bad output')
    if sha(a.source)!=source_hash or sha(a.manifest)!=manifest_hash or sha(model_path)!=model_hash:raise RuntimeError('input changed')
    artifact=a.out/'human_20hz_declared.npz'
    with artifact.open('xb') as f:np.savez_compressed(f,human=pts,source_frames=ix,source_times=ix.astype(np.float64)/fps)
    report={'id':a.id,'source_sha256':source_hash,'manifest_sha256':manifest_hash,'model_sha256':model_hash,'script_sha256':sha(Path(__file__)),'artifact_sha256':sha(artifact),'declared_fps':fps,'source_frames':n,'sampling_stride':6,'output_points':len(ix),'last_source_frame':int(ix[-1]),'nominal_humanml_interval':case['nominal_indices'],'source_type':'SMPL-X','training_ready':False,'limitations':['Full source clip at declared FPS, not an official HumanML time slice.','Physical clock and source identity unresolved; caption semantics not accepted.','Stride sampling has no anti-alias filtering.','Locked-head SMPL-X with zero hands and face; not official SMPL+H.','Human points only; no robot or physical acceptance.']}
    with (a.out/'report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()

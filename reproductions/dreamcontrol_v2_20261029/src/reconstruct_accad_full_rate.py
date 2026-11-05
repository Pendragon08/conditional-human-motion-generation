"""Reconstruct one complete candidate excerpt at its declared 120 Hz.

This preserves the requested human-to-robot-before-20-Hz ordering. Source
identity, time, SMPL-X equivalence and annotation semantics are not accepted.
"""
import contextlib
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import smplx
import torch

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
HERE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    ranked=ROOT/'short_independent_candidates_20261031/ranked.json'
    selection=ROOT/'independent_amass_selection_v2_20261031/candidates.json'
    rank=json.loads(ranked.read_text()); catalog=json.loads(selection.read_text())
    if sha(selection)!=rank['selection_sha256']:raise ValueError('selection changed')
    item=next(x for x in rank['candidates'] if x['id']=='013267')
    if item['parent_group'] not in {x['parent_group'] for x in catalog['selected']}:
        raise ValueError('unknown training source group')
    src=Path(item['candidate_path']);source_sha=sha(src)
    with np.load(src,allow_pickle=False) as d:
        pose=d['poses'].copy();trans=d['trans'].copy();betas=d['betas'].copy()
        fpskey=next((key for key in ('mocap_framerate','mocap_frame_rate') if key in d.files),None)
        if fpskey is None:raise ValueError('missing source FPS')
        fps=float(d[fpskey].item());gender=str(d['gender'].item())
    if fps!=120 or pose.ndim!=2 or pose.shape[1]!=165 or trans.shape!=(len(pose),3) or betas.shape!=(16,):
        raise ValueError('unexpected candidate format')
    if gender not in ('male','female','neutral') or item['start']!=0 or item['end']!=60:
        raise ValueError('unexpected clip metadata')
    frames=np.arange(item['start']*6,(item['end']-1)*6+1,dtype=np.int64)
    if len(frames)!=355 or frames[-1]>=len(pose):raise ValueError('source clip outside file')
    with contextlib.redirect_stdout(sys.stderr):
        model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=16,
                           use_pca=False,batch_size=1,ext='npz').cpu()
    chunks=[]
    for i in range(0,len(frames),60):
        ix=frames[i:i+60];n=len(ix)
        conv=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32)
        zeros=lambda width:torch.zeros((n,width),dtype=torch.float32)
        with torch.no_grad():
            p=model(betas=conv(np.broadcast_to(betas,(n,16)).copy()),global_orient=conv(pose[ix,:3]),
                    body_pose=conv(pose[ix,3:66]),transl=conv(trans[ix]),left_hand_pose=zeros(45),
                    right_hand_pose=zeros(45),jaw_pose=zeros(3),leye_pose=zeros(3),reye_pose=zeros(3),
                    expression=zeros(model.num_expression_coeffs),return_verts=False).joints[:,:22].numpy()
        chunks.append(p)
    points=np.concatenate(chunks)
    if points.shape!=(355,22,3) or not np.isfinite(points).all() or sha(src)!=source_sha:
        raise ValueError('invalid reconstruction or changed source')
    artifact=HERE/'human_120hz.npz'
    with artifact.open('xb') as f:
        np.savez_compressed(f,human=points,source_frames=frames,source_times=frames.astype(np.float64)/fps)
    modelpath=MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz')
    result={'id':item['id'],'source_path':item['source_path'],'source_parent':item['parent_group'],
            'source_candidate_path':str(src),'source_sha256':source_sha,'ranked_sha256':sha(ranked),
            'model_sha256':sha(modelpath),'gender':gender,'fps_declared':fps,'frames':len(frames),
            'first_last_raw_frames':[int(frames[0]),int(frames[-1])],
            'caption_member_sha256':item['caption_member_sha256'],'captions':item['captions'],
            'artifact_sha256':sha(artifact),'script_sha256':sha(Path(__file__)),'training_ready':False,
            'limitations':['source path match and 120Hz header not independently verified',
                           'locked-head SMPL-X 22-point approximation with zero hands/face',
                           'human/robot transition, real physics and full-clip caption semantics unaccepted']}
    with (HERE/'human_report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

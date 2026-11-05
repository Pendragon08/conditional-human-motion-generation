"""Candidate-source overlap under approximate SMPL-X and 263D re-encoding."""
import contextlib
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import smplx
import torch
from feature263 import encode
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
SOURCE=ROOT/'bmlhandball_overlap_witness_v2_20261030/report.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def statistics(a,b):
    d=np.abs(a-b);out={}
    for name,start,end in [('root',0,4),('ric',4,67),('rotation6d',67,193),('local_velocity',193,259),('foot_bits',259,263)]:
        x=d[:,start:end];out[name]={'max_abs':float(x.max()),'mean_abs':float(x.mean()),'nonzero_rows_gt_1e-6':int(np.any(x>1e-6,axis=1).sum())}
    out['feature_rows_exact_equal']=int(np.all(a==b,axis=1).sum());out['all_features_max_abs']=float(d.max());return out
def main():
    torch.set_num_threads(1);evidence=json.loads(SOURCE.read_text());rows=[];arrays={}
    for case in evidence['witnesses']:
        src=Path(case['candidate_path']);assert sha(src)==case['candidate_sha256']; ids=list(case['index_slices']); (a,b)=[case['index_slices'][k] for k in ids]
        with np.load(src,allow_pickle=False) as f:pose=np.asarray(f['poses']);trans=np.asarray(f['trans']);betas=np.asarray(f['betas']);gender=str(f['gender'].item());fps=float(f['mocap_framerate'].item())
        assert pose.shape[1]==156 and fps==120 and betas.shape==(16,) and gender in ('male','female','neutral')
        modelpath=MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz'); sample=np.arange(0,len(pose),6)
        def reconstruct(bounds):
            frames=sample[bounds[0]:bounds[1]];T=len(frames)
            with contextlib.redirect_stdout(sys.stderr):
                model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=16,use_pca=False,batch_size=T,ext='npz').cpu()
                conv=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32);zeros=lambda n:torch.zeros((T,n),dtype=torch.float32)
                with torch.no_grad():j=model(betas=conv(np.broadcast_to(betas,(T,16)).copy()),global_orient=conv(pose[frames,:3]),body_pose=conv(pose[frames,3:66]),transl=conv(trans[frames]),left_hand_pose=zeros(45),right_hand_pose=zeros(45),jaw_pose=zeros(3),leye_pose=zeros(3),reye_pose=zeros(3),expression=zeros(model.num_expression_coeffs),return_verts=False).joints[:,:22].numpy()
            return frames,j
        fa,ja=reconstruct(a);fb,jb=reconstruct(b)
        start=max(a[0],b[0]);end=min(a[1],b[1]);shared=end-start; assert shared==case['shared_sample_count']
        ja_shared=ja[start-a[0]:end-a[0]];jb_shared=jb[start-b[0]:end-b[0]]
        da=encode(ja,foot_velocity_squared_threshold=.002);db=encode(jb,foot_velocity_squared_threshold=.002)
        last=min(a[1]-1,b[1]-1);A=da[start-a[0]:last-a[0]];B=db[start-b[0]:last-b[0]];assert len(A)==len(B)==shared-1
        name='case_'+str(len(rows));arrays[name+'_joints_a']=ja_shared;arrays[name+'_joints_b']=jb_shared;arrays[name+'_features_a']=A;arrays[name+'_features_b']=B;arrays[name+'_source_frames']=sample[start:end]
        rows.append({'ids':ids,'index_slices':case['index_slices'],'candidate_sha256':sha(src),'model_sha256':sha(modelpath),'gender':gender,'declared_source_fps':fps,'stride':6,'shared_raw_joint_frames':shared,'shared_feature_rows':len(A),'shared_frame_first_last':[int(sample[start]),int(sample[end-1])],'independent_fk_overlap_max_component_m':float(np.max(np.abs(ja_shared-jb_shared))),'encoded_overlap':statistics(A,B)})
    artifact=Path(sys.argv[1]);
    with artifact.open('xb') as f:np.savez_compressed(f,**arrays)
    out={'candidate_overlap_report_sha256':sha(SOURCE),'encoder_sha256':sha(Path(sys.modules['feature263'].__file__)),'script_sha256':sha(Path(__file__)),'artifact_sha256':sha(artifact),'cases':rows,'limitations':'Two nonrandom candidate BMLhandball files and index mappings; 120 Hz is declared, not measured. SMPL-X betas/body transfer from 156D AMASS poses is an unvalidated approximation to official SMPL+H, hands/face set zero. Independently reconstructed candidate overlap and separately canonicalized/smoothed 263D are not official published features, actual identity or physical data; no train split changed or data accepted.'}
    with Path(sys.argv[2]).open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
if __name__=='__main__':main()

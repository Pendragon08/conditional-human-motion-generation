"""120-to-20 Hz joint-filter witness on two candidate BMLhandball sources."""
import contextlib
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import smplx
import torch
from scipy.signal import firwin,resample_poly
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
SOURCE=ROOT/'bmlhandball_overlap_witness_v2_20261030/report.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(1);evidence=json.loads(SOURCE.read_text());weights=firwin(121,1/6,window=('kaiser',5.0));array={};rows=[]
    for index,case in enumerate(evidence['witnesses']):
        src=Path(case['candidate_path']);assert sha(src)==case['candidate_sha256']
        with np.load(src,allow_pickle=False) as f:pose=np.asarray(f['poses']);trans=np.asarray(f['trans']);betas=np.asarray(f['betas']);gender=str(f['gender'].item());fps=float(f['mocap_framerate'].item())
        assert fps==120 and pose.shape[1]==156 and betas.shape==(16,)
        start,end=case['shared_sampled_indices'];first=(start-10)*6;last=(end-1+10)*6
        assert first>=0 and last<len(pose) and (last-first)%6==0
        with contextlib.redirect_stdout(sys.stderr):model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=16,use_pca=False,batch_size=1,ext='npz').cpu()
        parents=model.parents[:22].detach().cpu().numpy().astype(int);parts=[]
        for base in range(first,last+1,60):
            frames=np.arange(base,min(base+60,last+1));n=len(frames);cast=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32);zeros=lambda d:torch.zeros((n,d),dtype=torch.float32)
            with torch.no_grad():out=model(betas=cast(np.broadcast_to(betas,(n,16)).copy()),global_orient=cast(pose[frames,:3]),body_pose=cast(pose[frames,3:66]),transl=cast(trans[frames]),left_hand_pose=zeros(45),right_hand_pose=zeros(45),jaw_pose=zeros(3),leye_pose=zeros(3),reye_pose=zeros(3),expression=zeros(model.num_expression_coeffs),return_verts=False).joints[:,:22].numpy()
            parts.append(out)
        all_points=np.concatenate(parts).astype(np.float64);assert len(all_points)==last-first+1
        filtered=resample_poly(all_points,1,6,axis=0,window=weights,padtype='line');count=end-start;raw=all_points[60:60+count*6:6];aa=filtered[10:10+count];assert len(raw)==len(aa)==case['shared_sample_count']
        delta=np.linalg.norm(raw-aa,axis=2);raw_bones=np.linalg.norm(raw[:,1:]-raw[:,parents[1:]],axis=2);aa_bones=np.linalg.norm(aa[:,1:]-aa[:,parents[1:]],axis=2);bones=np.abs(raw_bones-aa_bones)
        prefix='case_'+str(index);array[prefix+'_points120']=all_points;array[prefix+'_decimated']=raw;array[prefix+'_filtered']=aa;array[prefix+'_parents']=parents
        rows.append({'ids':list(case['index_slices']),'candidate_path':str(src),'candidate_sha256':sha(src),'model_sha256':sha(MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz')),'gender':gender,'declared_fps':fps,'computed_raw_frames_inclusive':[first,last],'selected_first_last_raw_frames':[start*6,(end-1)*6],'selected_samples':count,'filter_margin_each_side_source_frames':60,'mean_joint_position_difference_m':float(delta.mean()),'max_joint_position_difference_m':float(delta.max()),'mean_abs_bone_length_difference_m':float(bones.mean()),'max_abs_bone_length_difference_m':float(bones.max())})
    artifact=Path(sys.argv[1]);
    with artifact.open('xb') as f:np.savez_compressed(f,**array,filter_coefficients=weights)
    result={'source_report_sha256':sha(SOURCE),'script_sha256':sha(Path(__file__)),'artifact_sha256':sha(artifact),'filter_taps':len(weights),'cutoff_cycles_per_source_sample':1/12,'window':'kaiser beta=5','filter_phase':'centered noncausal','cases':rows,'limitations':'Two nonrandom BMLhandball candidates; 120 Hz header not independently measured. Source pose/betas to locked-head SMPL-X are an unverified SMPL+H approximation; zero hand/face, first 22 joints only. Symmetric filtering uses future frames, is not the author method or training data; filtering changes bone lengths. Distances compare methods, not actual accuracy, posture, contact or dynamics.'}
    with Path(sys.argv[2]).open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
if __name__=='__main__':main()

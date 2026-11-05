"""Independent direct FIR and selected source-frame reconstruction checks."""
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
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(1);report=Path(sys.argv[1]);r=json.loads(report.read_text());p=report.with_name('witness.npz');assert sha(p)==r['artifact_sha256']
    assert sha(ROOT/'bmlhandball_overlap_witness_v2_20261030/report.json')==r['source_report_sha256']
    grid=np.arange(-60,61,dtype=float);weights=np.sinc(grid/6)/6*np.kaiser(121,5);weights/=weights.sum();max_filter=0.;max_forward=0.
    with np.load(p,allow_pickle=False) as data:
        assert np.max(np.abs(weights-data['filter_coefficients']))<1e-14
        for i,row in enumerate(r['cases']):
            src=Path(row['candidate_path']);assert sha(src)==row['candidate_sha256'];points=data[f'case_{i}_points120'];raw=data[f'case_{i}_decimated'];aa=data[f'case_{i}_filtered'];parents=data[f'case_{i}_parents'];count=row['selected_samples'];assert raw.shape==aa.shape==(count,22,3)
            first,last=row['computed_raw_frames_inclusive'];assert len(points)==last-first+1 and np.array_equal(raw,points[60:60+count*6:6])
            d=np.linalg.norm(raw-aa,axis=2);bone=np.abs(np.linalg.norm(raw[:,1:]-raw[:,parents[1:]],axis=2)-np.linalg.norm(aa[:,1:]-aa[:,parents[1:]],axis=2))
            assert abs(float(d.mean())-row['mean_joint_position_difference_m'])<1e-12 and abs(float(d.max())-row['max_joint_position_difference_m'])<1e-12
            assert abs(float(bone.mean())-row['mean_abs_bone_length_difference_m'])<1e-12 and abs(float(bone.max())-row['max_abs_bone_length_difference_m'])<1e-12
            select=[0,count//2,count-1]
            for k in select:
                center=60+6*k;independent=np.einsum('t,tjk->jk',weights,points[center-60:center+61]);err=float(np.max(np.abs(independent-aa[k])));max_filter=max(max_filter,err);assert err<1e-10
            frames=np.array([first+60+6*k for k in select]);assert frames[0]==row['selected_first_last_raw_frames'][0] and frames[-1]==row['selected_first_last_raw_frames'][1]
            with np.load(src,allow_pickle=False) as f:pose=f['poses'][frames];trans=f['trans'][frames];betas=f['betas'];gender=str(f['gender'].item());fps=float(f['mocap_framerate'])
            assert fps==row['declared_fps']==120 and sha(MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz'))==row['model_sha256']
            with contextlib.redirect_stdout(sys.stderr):
                model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=16,use_pca=False,batch_size=3,ext='npz').cpu();cast=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32);zeros=lambda d:torch.zeros((3,d))
                with torch.no_grad():recalc=model(betas=cast(np.broadcast_to(betas,(3,16)).copy()),global_orient=cast(pose[:,:3]),body_pose=cast(pose[:,3:66]),transl=cast(trans),left_hand_pose=zeros(45),right_hand_pose=zeros(45),jaw_pose=zeros(3),leye_pose=zeros(3),reye_pose=zeros(3),expression=zeros(model.num_expression_coeffs),return_verts=False).joints[:,:22].numpy()
            err=float(np.max(np.abs(recalc-raw[select])));max_forward=max(max_forward,err);assert err<2e-6
    out={'status':'passed','witness_sha256':sha(report),'artifact_sha256':sha(p),'script_sha256':sha(Path(__file__)),'cases_checked':len(r['cases']),'selected_samples_reconstructed':len(r['cases'])*3,'max_fir_direct_sum_component_error_m':max_filter,'max_smplx_component_error_m':max_forward,'limitations':'Independent sinc/Kaiser direct sum of saved joint trajectories and same-library selected frame checks, not all original frames, real FPS, SMPL+H equivalence or real motion accuracy.'}
    with Path(sys.argv[2]).open('x',encoding='utf8') as f:json.dump(out,f,indent=2)
if __name__=='__main__':main()

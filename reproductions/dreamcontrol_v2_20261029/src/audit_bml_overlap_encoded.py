"""Recheck candidate provenance, split indexes and selected SMPL-X forward frames."""
import contextlib
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import smplx
import torch
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');PUBLIC=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public')
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead');OLD=ROOT/'bmlhandball_overlap_witness_v2_20261030/report.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(1);report=Path(sys.argv[1]);r=json.loads(report.read_text());artifact=report.with_name('overlap.npz');assert sha(artifact)==r['artifact_sha256'] and sha(OLD)==r['candidate_overlap_report_sha256']
    old=json.loads(OLD.read_text());index=PUBLIC/'index.csv';assert sha(index)==old['official_index_sha256']
    with index.open(newline='',encoding='utf8') as f:idx={row['new_name'].removesuffix('.npy'):row for row in csv.DictReader(f)}
    splits={x:set((PUBLIC/'HumanML3D'/f'{x}.txt').read_text().splitlines()) for x in ('train','val','test')};maximum=0.;groups=[('root',0,4),('ric',4,67),('rotation6d',67,193),('local_velocity',193,259),('foot_bits',259,263)]
    with np.load(artifact,allow_pickle=False) as saved:
        for i,(row,prior) in enumerate(zip(r['cases'],old['witnesses'])):
            assert row['ids']==list(prior['index_slices']) and row['index_slices']==prior['index_slices']
            for key,split in prior['ids'].items():
                assert key in splits[split] and idx[key]['source_path']==prior['source_path']
                assert [int(idx[key]['start_frame']),int(idx[key]['end_frame'])]==prior['index_slices'][key]
            src=Path(prior['candidate_path']);assert sha(src)==row['candidate_sha256']==prior['candidate_sha256']
            start=max(x[0] for x in prior['index_slices'].values());end=min(x[1] for x in prior['index_slices'].values());frames=saved[f'case_{i}_source_frames']
            assert np.array_equal(frames,np.arange(start,end)*6) and len(frames)==row['shared_raw_joint_frames']
            a=saved[f'case_{i}_joints_a'];b=saved[f'case_{i}_joints_b'];A=saved[f'case_{i}_features_a'];B=saved[f'case_{i}_features_b']
            assert a.shape==b.shape==(len(frames),22,3) and A.shape==B.shape==(len(frames)-1,263)
            assert float(np.max(np.abs(a-b)))==row['independent_fk_overlap_max_component_m']
            delta=np.abs(A-B)
            for name,left,right in groups:
                q=delta[:,left:right];rec=row['encoded_overlap'][name];assert float(q.max())==rec['max_abs'] and abs(float(q.mean())-rec['mean_abs'])<1e-12 and int(np.any(q>1e-6,axis=1).sum())==rec['nonzero_rows_gt_1e-6']
            assert int(np.all(A==B,axis=1).sum())==row['encoded_overlap']['feature_rows_exact_equal']
            take=np.array([0,len(frames)//2,len(frames)-1]);selected=frames[take]
            with np.load(src,allow_pickle=False) as data:pose=data['poses'][selected];trans=data['trans'][selected];beta=data['betas'];gender=str(data['gender'].item());fps=float(data['mocap_framerate'])
            assert fps==120 and sha(MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz'))==row['model_sha256']
            with contextlib.redirect_stdout(sys.stderr):
                m=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=16,use_pca=False,batch_size=3,ext='npz').cpu();cast=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32);zero=lambda n:torch.zeros((3,n))
                with torch.no_grad():j=m(betas=cast(np.broadcast_to(beta,(3,16)).copy()),global_orient=cast(pose[:,:3]),body_pose=cast(pose[:,3:66]),transl=cast(trans),left_hand_pose=zero(45),right_hand_pose=zero(45),jaw_pose=zero(3),leye_pose=zero(3),reye_pose=zero(3),expression=zero(m.num_expression_coeffs),return_verts=False).joints[:,:22].numpy()
            err=float(np.max(np.abs(j-a[take])));maximum=max(maximum,err);assert err<2e-6
    out={'status':'passed','witness_sha256':sha(report),'artifact_sha256':sha(artifact),'official_index_sha256':sha(index),'script_sha256':sha(Path(__file__)),'cases_checked':len(r['cases']),'selected_smplx_frames_checked':len(r['cases'])*3,'max_smplx_coordinate_difference_m':maximum,'limitations':'Checks SMPL-X with same library on selected source frames, not official SMPL+H or intermediate poses; encoded statistics recomputed from saved arrays rather than independently re-encoding. No actual FPS, license or physical acceptance.'}
    with Path(sys.argv[2]).open('x',encoding='utf8') as f:json.dump(out,f,indent=2)
if __name__=='__main__':main()

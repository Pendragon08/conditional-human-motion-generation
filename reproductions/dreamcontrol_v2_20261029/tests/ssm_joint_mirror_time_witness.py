"""CPU-only mirror/time witness on two marker-aligned SSM candidates."""
import contextlib,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch,smplx
from humanml_joint_mirroring import amass_to_humanml,mirror_humanml
from humanml_time_mapping import aligned_grid

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');OUT=Path(__file__).resolve().parent
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
ALIGN=ROOT/'ssm_id_alignment_20261030/ids.jsonl'
EXPECTED='575c9c008298dbf30d21317f258346a3dd11cc7efc6858d242a26c11c6333b46'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def independent_official_mirror(a):
    # Separate direct-index implementation of the pinned official notebook.
    out=a.copy();out[:,:,0]*=-1
    left=[1,4,7,10,13,16,18,20];right=[2,5,8,11,14,17,19,21]
    temp=out[:,right,:].copy();out[:,right,:]=out[:,left,:];out[:,left,:]=temp
    return out

def main():
    torch.set_num_threads(1)
    if sha(ALIGN)!=EXPECTED:raise ValueError('alignment receipt changed')
    official=ROOT/'humanml_official_semantics_20261030/raw_pose_processing.ipynb'
    if sha(official)!='fc1ffb4ed2f0e808d94fd434a2e75a02099aceed531b35dd4fae7c9e068ffe6f':raise ValueError('official notebook changed')
    records={r['motion_id']:r for r in map(json.loads,ALIGN.read_text().splitlines())}
    results=[]
    for name in ('000008','002511'):
        row=records[name];mirror=records['M'+name]
        if row['mirrored'] or not mirror['mirrored'] or row['source_path']!=mirror['source_path'] or (row['start'],row['stop'])!=(mirror['start'],mirror['stop']):
            raise ValueError('source mirror correspondence mismatch')
        rec_path=Path(row['source_receipt']);rec=json.loads(rec_path.read_text())
        target=Path(rec['smplx_file']);reference=Path(rec['reference_file'])
        if sha(target)!=rec['smplx_sha256'] or sha(reference)!=rec['reference_sha256'] or rec['timeline']['status']!='FRAME_IDENTITY_VERIFIED_BY_MEASURED_MARKERS':
            raise ValueError('source evidence mismatch')
        with np.load(reference,allow_pickle=False) as f:
            fps=float(f['mocap_framerate']);raw_count=len(f['trans'])
        if fps!=row['reference_fps'] or raw_count!=row['reference_raw_frame_count']:raise ValueError('reference timing changed')
        grid=aligned_grid(raw_count,fps,row['start'],row['stop']);ind=np.asarray(grid['source_indices'],dtype=int)
        with np.load(target,allow_pickle=False) as f:
            trans=np.asarray(f['trans'])[ind];orient=np.asarray(f['root_orient'])[ind];body=np.asarray(f['pose_body'])[ind]
            betas=np.asarray(f['betas'],dtype=np.float32).reshape(-1)[:16];gender=str(f['gender'].item())
        if len(betas)!=16 or gender not in ('male','female','neutral'):raise ValueError('unexpected model input')
        n=len(ind);tensor=lambda x:torch.as_tensor(x,dtype=torch.float32);zeros=lambda m:torch.zeros((n,m),dtype=torch.float32)
        with contextlib.redirect_stdout(sys.stderr):
            model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=16,use_pca=False,batch_size=n,ext='npz').cpu()
            with torch.no_grad():
                result=model(betas=tensor(np.broadcast_to(betas,(n,16)).copy()),global_orient=tensor(orient),body_pose=tensor(body),transl=tensor(trans),
                    left_hand_pose=zeros(45),right_hand_pose=zeros(45),jaw_pose=zeros(3),leye_pose=zeros(3),reye_pose=zeros(3),
                    expression=zeros(model.num_expression_coeffs),return_verts=False)
                joints=result.joints[:,:22].cpu().numpy().astype(np.float64)
        normal=amass_to_humanml(joints);mirrored=mirror_humanml(normal);independent=independent_official_mirror(normal)
        if not np.array_equal(mirrored,independent) or not np.array_equal(mirror_humanml(mirrored),normal):raise ValueError('mirror differs')
        # Linear interpolation of joint positions is a numerical witness, not robot or author preprocessing.
        slot={int(frame):i for i,frame in enumerate(ind)};interp=[];mirror_interp=[]
        for w in grid['weights']:
            a=slot[w['left_frame']];b=slot[w['right_frame']];alpha=w['alpha']
            interp.append(normal[a]*(1-alpha)+normal[b]*alpha)
            mirror_interp.append(mirrored[a]*(1-alpha)+mirrored[b]*alpha)
        interp=np.asarray(interp);mirror_interp=np.asarray(mirror_interp)
        commutation=float(np.max(np.abs(mirror_interp-mirror_humanml(interp))))
        if commutation>1e-10:raise ValueError('mirror/time interpolation mismatch')
        artifact=OUT/(name+'_joint_witness.npz')
        with artifact.open('xb') as f:np.savez_compressed(f,source_frames=ind,source_times=ind/fps,
            original_joints_humanml=normal,mirrored_joints_humanml=mirrored,
            output_times=np.array([w['time'] for w in grid['weights']]),
            original_joint_linear_20hz=interp,mirrored_joint_linear_20hz=mirror_interp)
        results.append({'id':name,'mirror_id':'M'+name,'source_sha256':sha(target),'reference_sha256':sha(reference),
                        'source_receipt_sha256':sha(rec_path),'model_sha256':sha(MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz')),
                        'input_samples':n,'input_fps_actual_from_reference':grid['actual_reference_sample_fps'],
                        'output_samples':len(interp),'source_frame_first_last':[int(ind[0]),int(ind[-1])],
                        'source_first_last_duration_s':grid['source_duration_s'],
                        'mirror_independent_max_difference_m':float(np.max(np.abs(mirrored-independent))),
                        'mirror_involution_max_difference_m':float(np.max(np.abs(mirror_humanml(mirrored)-normal))),
                        'linear_resample_mirror_commutation_max_difference_m':commutation,'artifact_sha256':sha(artifact)})
        print(json.dumps({'completed':name,'source_samples':n,'output_samples':len(interp)}),flush=True)
    out={'script_sha256':sha(__file__),'mirror_module_sha256':sha(OUT/'humanml_joint_mirroring.py'),
         'time_module_sha256':sha(OUT/'humanml_time_mapping.py'),'alignment_sha256':sha(ALIGN),'official_notebook_sha256':sha(official),'witnesses':results,
         'limitations':'Two marker-aligned SSM candidates with existing locked-head SMPL-X, zero hand/face poses; not verified equivalent to official SMPL+H or actual published features. Reuses old marker evidence, does not remeasure. Linear interpolation of reconstructed joints is diagnostic and has no anti-aliasing; no rotational interpolation here. Mirror is positions only, no caption left/right, object, robot retargeting, contact, physics, or training data.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()

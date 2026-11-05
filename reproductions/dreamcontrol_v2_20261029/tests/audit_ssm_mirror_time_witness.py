"""Independent source and numeric check of two SSM mirror/time witnesses."""
import contextlib,hashlib,json,sys
from pathlib import Path
import numpy as np
import smplx,torch
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');OUT=Path(__file__).resolve().parent
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
ALIGN=ROOT/'ssm_id_alignment_20261030/ids.jsonl'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    torch.set_num_threads(1);report=json.loads((OUT/'report.json').read_text());rows={r['motion_id']:r for r in map(json.loads,ALIGN.read_text().splitlines())}
    for p,k in ((ALIGN,'alignment_sha256'),(OUT/'ssm_joint_mirror_time_witness.py','script_sha256'),(OUT/'humanml_joint_mirroring.py','mirror_module_sha256'),(OUT/'humanml_time_mapping.py','time_module_sha256')):
        if sha(p)!=report[k]:raise ValueError('provenance mismatch')
    checks=[]
    for item in report['witnesses']:
        name=item['id'];row=rows[name];receipt=Path(row['source_receipt']);r=json.loads(receipt.read_text())
        target=Path(r['smplx_file']);reference=Path(r['reference_file']);artifact=OUT/(name+'_joint_witness.npz')
        for p,expected in ((receipt,item['source_receipt_sha256']),(target,item['source_sha256']),(reference,item['reference_sha256']),(artifact,item['artifact_sha256'])):
            if sha(p)!=expected:raise ValueError('input mismatch')
        with np.load(reference,allow_pickle=False) as d:fps=float(d['mocap_framerate']);count=len(d['trans'])
        with np.load(artifact,allow_pickle=False) as d:
            ind=d['source_frames'];times=d['source_times'];normal=d['original_joints_humanml'];mirrored=d['mirrored_joints_humanml']
            outtimes=d['output_times'];interp=d['original_joint_linear_20hz'];mirror_interp=d['mirrored_joint_linear_20hz']
        expected=np.arange(0,count,int(fps/20))[row['start']:row['stop']]
        if not np.array_equal(ind,expected) or not np.allclose(times,ind/fps,rtol=0,atol=1e-12):raise ValueError('time mismatch')
        # Direct permutation/matrix independently checks source mirror.
        perm=np.arange(22);left=[1,4,7,10,13,16,18,20];right=[2,5,8,11,14,17,19,21]
        perm[left]=right;perm[right]=left
        expected_mirror=normal[:,perm,:]*np.array([-1.,1.,1.])
        if not np.array_equal(expected_mirror,mirrored):raise ValueError('mirror mismatch')
        expected_interp=np.stack([np.interp(outtimes,times-times[0],normal.reshape(len(times),-1)[:,k]) for k in range(66)],axis=-1).reshape(-1,22,3)
        err_interp=float(np.max(np.abs(expected_interp-interp)))
        expected_mirror_interp=interp[:,perm,:]*np.array([-1.,1.,1.])
        err_commute=float(np.max(np.abs(expected_mirror_interp-mirror_interp)))
        if err_interp>1e-9 or err_commute>1e-9:raise ValueError('resampled positions mismatch')
        pick=np.array([0,len(ind)//2,len(ind)-1]);src_idx=ind[pick]
        with np.load(target,allow_pickle=False) as d:
            trans=d['trans'][src_idx];orient=d['root_orient'][src_idx];body=d['pose_body'][src_idx];betas=np.asarray(d['betas'],dtype=np.float32).reshape(-1)[:16];gender=str(d['gender'].item())
        modelpath=MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz')
        if sha(modelpath)!=item['model_sha256']:raise ValueError('model changed')
        with contextlib.redirect_stdout(sys.stderr):
            model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=16,use_pca=False,batch_size=3,ext='npz').cpu()
            tensor=lambda v:torch.as_tensor(v,dtype=torch.float32);zero=lambda m:torch.zeros((3,m),dtype=torch.float32)
            with torch.no_grad():
                out=model(betas=tensor(np.broadcast_to(betas,(3,16)).copy()),global_orient=tensor(orient),body_pose=tensor(body),transl=tensor(trans),
                          left_hand_pose=zero(45),right_hand_pose=zero(45),jaw_pose=zero(3),leye_pose=zero(3),reye_pose=zero(3),
                          expression=zero(model.num_expression_coeffs),return_verts=False)
                original=out.joints[:,:22].cpu().numpy().astype(np.float64)
        independent=original[:,:,[0,2,1]].copy();independent[:,:,0]*=-1
        err_source=float(np.max(np.abs(independent-normal[pick])))
        if err_source>2e-5:raise ValueError('source FK mismatch')
        checks.append({'id':name,'verified_sampled_frames':src_idx.tolist(),'source_fk_max_abs_diff_m':err_source,
                       'independent_interp_max_abs_diff_m':err_interp,'mirror_interp_max_abs_diff_m':err_commute,
                       'final_20hz_grid_tail_s':float(times[-1]-times[0]-outtimes[-1])})
    result={'script_sha256':sha(__file__),'witness_report_sha256':sha(OUT/'report.json'),'checks':checks,
            'limitations':'Independent arithmetic and three-frame source-model checks per recording; same approximate locked-head model and prior marker evidence. Joint-space linear interpolation lacks anti-aliasing; mirrored captions, rotation, robot retargeting, physics and training acceptance not checked.'}
    with (OUT/'independent_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()

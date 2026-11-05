"""Independent matrix-conjugation check of pose-reflection diagnostic."""
import contextlib,hashlib,json,sys
from pathlib import Path
import numpy as np,torch,smplx
from scipy.spatial.transform import Rotation
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');BASE=ROOT/'humanml_mirror_time_witness_v2_20261030'
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
OUT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    torch.set_num_threads(1);prior=json.loads((OUT/'report.json').read_text());assert sha(OUT/'run.py')==prior['script_sha256']
    records={x['motion_id']:x for x in map(json.loads,(ROOT/'ssm_id_alignment_20261030/ids.jsonl').read_text().splitlines())};out=[]
    left=[1,4,7,10,13,16,18,20];right=[2,5,8,11,14,17,19,21];M=np.diag([-1.,1.,1.])
    for row in prior['rows']:
        ident=row['id'];rec=json.loads(Path(records[ident]['source_receipt']).read_text());path=Path(rec['smplx_file']);witness=BASE/(ident+'_joint_witness.npz')
        assert sha(path)==row['source_sha256'] and sha(witness)==row['witness_sha256']
        with np.load(witness,allow_pickle=False) as d:
            ix=[0,len(d['source_frames'])//2,len(d['source_frames'])-1];frames=d['source_frames'][ix];reference=d['original_joints_humanml'][ix]
        assert [int(i) for i in frames]==row['source_frames']
        with np.load(path,allow_pickle=False) as d:
            pos=np.array(d['trans'][frames]);rot=np.array(d['root_orient'][frames]);body=np.array(d['pose_body'][frames]);betas=np.asarray(d['betas'],dtype=np.float32).reshape(-1)[:16];gender=str(d['gender'].item())
        assert sha(MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz'))==row['model_sha256']
        angles=np.concatenate([rot[:,None,:],body.reshape(3,21,3)],axis=1);matrix=Rotation.from_rotvec(angles.reshape(-1,3)).as_matrix().reshape(3,22,3,3)
        reflected=M@matrix@M; swapped=reflected.copy();swapped[:,left]=reflected[:,right];swapped[:,right]=reflected[:,left]
        new_angles=Rotation.from_matrix(swapped.reshape(-1,3,3)).as_rotvec().reshape(3,22,3);new_pos=pos.copy();new_pos[:,0]*=-1
        with contextlib.redirect_stdout(sys.stderr):model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=16,use_pca=False,batch_size=3,ext='npz').cpu()
        T=lambda x:torch.as_tensor(x,dtype=torch.float32);zeros=lambda n:torch.zeros((3,n),dtype=torch.float32)
        with torch.no_grad():joints=model(betas=T(np.broadcast_to(betas,(3,16)).copy()),global_orient=T(new_angles[:,0]),body_pose=T(new_angles[:,1:].reshape(3,63)),transl=T(new_pos),left_hand_pose=zeros(45),right_hand_pose=zeros(45),jaw_pose=zeros(3),leye_pose=zeros(3),reye_pose=zeros(3),expression=zeros(model.num_expression_coeffs),return_verts=False).joints[:,:22].numpy().astype(np.float64)
        actual=joints[:,:,[0,2,1]].copy();actual[:,:,0]*=-1
        expected=reference.copy();expected[:,:,0]*=-1;temp=expected[:,left].copy();expected[:,left]=expected[:,right];expected[:,right]=temp
        diff=actual-expected;absolute=np.linalg.norm(diff,axis=-1);relative=np.linalg.norm(diff-diff[:,:1,:],axis=-1)
        checks={'world_point_error_mean_m':float(absolute.mean()),'world_point_error_max_m':float(absolute.max()),'pelvis_aligned_error_mean_m':float(relative.mean()),'pelvis_aligned_error_max_m':float(relative.max())}
        discrepancy=max(abs(row[k]-v) for k,v in checks.items());assert discrepancy<2e-6,(ident,discrepancy);out.append({'id':ident,'metric_difference_max_m':discrepancy})
    result={'rows':out,'primary_report_sha256':sha(OUT/'report.json'),'independent_script_sha256':sha(Path(__file__)),'limitations':'Reuses same SMPL-X forward model and two three-frame samples; independent rotation reflection and point mirror implementation, not independent physical truth.'}
    with (OUT/'independent_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()

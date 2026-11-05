"""Isolate reflected skeleton rest-shape effects; no source or training modifications."""
import contextlib,hashlib,json,sys
from pathlib import Path
import numpy as np,torch,smplx
from scipy.spatial.transform import Rotation
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MODELS=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
BASE=ROOT/'ssm_pose_reflection_witness_20261030';OUT=Path(__file__).resolve().parent
LEFT=[1,4,7,10,13,16,18,20];RIGHT=[2,5,8,11,14,17,19,21];X=np.diag([-1.,1.,1.]);PERM=np.arange(22);PERM[LEFT]=RIGHT;PERM[RIGHT]=LEFT
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def rest(model,beta):
    shaped=model.v_template + torch.einsum('vci,i->vc',model.shapedirs[:,:,:16],torch.tensor(beta,dtype=torch.float32))
    return (model.J_regressor@shaped)[:22].detach().numpy().astype(np.float64)
def fk(restj,rot,parents):
    n=len(rot);r=np.empty((n,22,3,3));p=np.empty((n,22,3));r[:,0]=rot[:,0];p[:,0]=restj[0]
    for j in range(1,22):
        q=int(parents[j]);p[:,j]=p[:,q]+np.einsum('nij,j->ni',r[:,q],restj[j]-restj[q]);r[:,j]=r[:,q]@rot[:,j]
    return p
def reflect(points):return (points[:,PERM,:]@X)
def main():
    torch.set_num_threads(1); previous=json.loads((BASE/'report.json').read_text());reference=previous['rows'][1]
    align=ROOT/'ssm_id_alignment_20261030/ids.jsonl';record=next(r for r in map(json.loads,align.read_text().splitlines()) if r['motion_id']=='002511')
    source=Path(json.loads(Path(record['source_receipt']).read_text())['smplx_file']);assert sha(source)==reference['source_sha256']
    with np.load(source,allow_pickle=False) as f:
        frames=np.array(reference['source_frames']);betas=np.asarray(f['betas'],dtype=np.float32).reshape(-1)[:16];root=f['root_orient'][frames];body=f['pose_body'][frames];trans=f['trans'][frames];gender=str(f['gender'].item())
    angles=np.concatenate([root[:,None,:],body.reshape(3,21,3)],axis=1);rot=Rotation.from_rotvec(angles.reshape(-1,3)).as_matrix().reshape(3,22,3,3)
    reflected_rot=(X@rot[:,PERM]@X);results=[]
    for sex in ('male','female','neutral'):
        modelpath=MODELS/'smplx'/('SMPLX_'+sex.upper()+'.npz')
        with contextlib.redirect_stdout(sys.stderr):model=smplx.create(str(MODELS),model_type='smplx',gender=sex,num_betas=16,use_pca=False,batch_size=3,ext='npz').cpu()
        parents=model.parents[:22].numpy();assert all(PERM[parents[j]]==parents[PERM[j]] for j in range(1,22))
        beta=betas if sex==gender else np.zeros(16,dtype=np.float32);j=rest(model,beta);center=j-j[0]
        restgap=np.linalg.norm(center-(center[PERM]@X),axis=1)
        original=fk(center,rot,parents);naive=fk(center,reflected_rot,parents)
        diff=np.linalg.norm(naive-reflect(original),axis=-1)
        sym=(center+(center[PERM]@X))/2
        symdiff=np.linalg.norm(fk(sym,reflected_rot,parents)-reflect(fk(sym,rot,parents)),axis=-1)
        assert symdiff.max()<1e-12
        rec={'gender':sex,'model_sha256':sha(modelpath),'betas':'002511 subject' if sex==gender else 'all zero',
             'rest_joint_asymmetry_mean_m':float(restgap.mean()),'rest_joint_asymmetry_max_m':float(restgap.max()),
             'posed_reflection_error_mean_m':float(diff.mean()),'posed_reflection_error_max_m':float(diff.max()),
             'artificial_symmetric_rest_error_max_m':float(symdiff.max())}
        if sex==gender:
            assert rec['model_sha256']==reference['model_sha256']
            assert abs(rec['posed_reflection_error_mean_m']-reference['pelvis_aligned_error_mean_m'])<2e-6
            assert abs(rec['posed_reflection_error_max_m']-reference['pelvis_aligned_error_max_m'])<2e-6
            # Verify manual kinematic positions against the actual SMPL-X forward model.
            T=lambda x:torch.as_tensor(x,dtype=torch.float32);zeros=lambda n:torch.zeros((3,n),dtype=torch.float32)
            with torch.no_grad():real=model(betas=T(np.broadcast_to(beta,(3,16)).copy()),global_orient=T(root),body_pose=T(body),transl=T(trans),left_hand_pose=zeros(45),right_hand_pose=zeros(45),jaw_pose=zeros(3),leye_pose=zeros(3),reye_pose=zeros(3),expression=zeros(model.num_expression_coeffs),return_verts=False).joints[:,:22].numpy()
            predicted=fk(j,rot,parents)+trans[:,None,:];rec['manual_fk_max_difference_m']=float(np.max(np.abs(real-predicted)));assert rec['manual_fk_max_difference_m']<2e-6
        results.append(rec)
    output={'source_sha256':sha(source),'previous_report_sha256':sha(BASE/'report.json'),'script_sha256':sha(Path(__file__)),'subject_frames':[int(v) for v in frames],
       'results':results,'limitations':'Same three poses and SSM beta as prior, plus opposite/neutral zero-beta templates; no independent subject or measured ground truth. Artificial skeleton symmetrization is diagnostic only, not a valid full SMPL-X model, not a proposed correction or training data. Does not prove other model versions or pose conventions.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps(output))
if __name__=='__main__':main()

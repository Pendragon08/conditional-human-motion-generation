"""Test a pose-space reflection hypothesis, without altering source or training data."""
import contextlib,hashlib,json,sys
from pathlib import Path
import numpy as np, torch, smplx
from scipy.spatial.transform import Rotation
from humanml_joint_mirroring import amass_to_humanml,mirror_humanml,BODY_LEFT,BODY_RIGHT

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
BASE=ROOT/'humanml_mirror_time_witness_v2_20261030'
ALIGN=ROOT/'ssm_id_alignment_20261030/ids.jsonl'
OUT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def reflect_pose(root,body):
    body=np.asarray(body).reshape(-1,21,3); allposes=np.concatenate([np.asarray(root)[:,None,:],body],axis=1)
    # Reflection through AMASS x=0 maps an axial rotation vector (x,y,z) to (x,-y,-z).
    transformed=allposes*np.array([1.,-1.,-1.]); result=transformed.copy()
    result[:,BODY_LEFT,:]=transformed[:,BODY_RIGHT,:];result[:,BODY_RIGHT,:]=transformed[:,BODY_LEFT,:]
    return result[:,0,:],result[:,1:,:].reshape(-1,63)
def main():
    torch.set_num_threads(1);reflection=np.diag([-1.,1.,1.])
    # Includes principal-axis and near-pi rotations; compare matrices, not rotation-vector branches.
    a=np.array([[.2,-.3,.5],[0.,0.,0.],[np.pi-1e-7,0.,0.],[0.,np.pi-1e-7,0.]])
    original=Rotation.from_rotvec(a).as_matrix(); transformed=Rotation.from_rotvec(a*[1,-1,-1]).as_matrix()
    formula=float(np.max(np.abs(transformed-reflection@original@reflection)));assert formula<1e-12
    prior=json.loads((BASE/'report.json').read_text()); byid={x['id']:x for x in prior['witnesses']}
    records={x['motion_id']:x for x in map(json.loads,ALIGN.read_text().splitlines())}; rows=[]
    for ident in ('000008','002511'):
        row=records[ident]; evidence=json.loads(Path(row['source_receipt']).read_text()); target=Path(evidence['smplx_file']); witness=BASE/(ident+'_joint_witness.npz')
        if sha(target)!=evidence['smplx_sha256'] or sha(witness)!=byid[ident]['artifact_sha256']:raise ValueError('source mismatch')
        with np.load(witness,allow_pickle=False) as d:
            inds=np.array([0,len(d['source_frames'])//2,len(d['source_frames'])-1]);frames=d['source_frames'][inds];existing=d['original_joints_humanml'][inds]
        with np.load(target,allow_pickle=False) as d:
            trans=np.array(d['trans'][frames]); root=np.array(d['root_orient'][frames]);body=np.array(d['pose_body'][frames]);betas=np.array(d['betas'],dtype=np.float32).reshape(-1)[:16];gender=str(d['gender'].item())
        modelpath=MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz')
        if sha(modelpath)!=byid[ident]['model_sha256']:raise ValueError('model mismatch')
        with contextlib.redirect_stdout(sys.stderr):model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=16,use_pca=False,batch_size=3,ext='npz').cpu()
        T=lambda x:torch.as_tensor(x,dtype=torch.float32); zeros=lambda n:torch.zeros((3,n),dtype=torch.float32)
        def forward(pos,rot,pose):
            with torch.no_grad():
                return model(betas=T(np.broadcast_to(betas,(3,16)).copy()),global_orient=T(rot),body_pose=T(pose),transl=T(pos),left_hand_pose=zeros(45),right_hand_pose=zeros(45),jaw_pose=zeros(3),leye_pose=zeros(3),reye_pose=zeros(3),expression=zeros(model.num_expression_coeffs),return_verts=False).joints[:,:22].numpy().astype(np.float64)
        base=amass_to_humanml(forward(trans,root,body));reference=mirror_humanml(base)
        mirrored_root,mirrored_body=reflect_pose(root,body);mirrored_trans=trans.copy();mirrored_trans[:,0]*=-1
        actual=amass_to_humanml(forward(mirrored_trans,mirrored_root,mirrored_body))
        baseerr=float(np.max(np.abs(base-existing)));assert baseerr<2e-6
        delta=actual-reference;absolute=np.linalg.norm(delta,axis=-1);relative=np.linalg.norm(delta-delta[:,:1,:],axis=-1)
        rows.append({'id':ident,'source_frames':[int(x) for x in frames],'source_sha256':sha(target),'model_sha256':sha(modelpath),'witness_sha256':sha(witness),
            'original_fk_max_component_difference_to_prior_m':baseerr,'world_point_error_mean_m':float(absolute.mean()),'world_point_error_max_m':float(absolute.max()),
            'pelvis_aligned_error_mean_m':float(relative.mean()),'pelvis_aligned_error_max_m':float(relative.max()),'pelvis_world_error_max_m':float(absolute[:,0].max())})
    result={'rows':rows,'rotation_matrix_algebra_max_difference':formula,'script_sha256':sha(Path(__file__)),'module_sha256':sha(OUT/'humanml_joint_mirroring.py'),
        'limitations':'Only a reflected-axis-angle hypothesis applied to three frames each with same SMPL-X betas and zero hand/face; official HumanML mirrors positions, not pose. Asymmetric template and joint conventions can cause residuals; no evidence of physical truth, model version identity, or source timing. No motion or training output.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))
if __name__=='__main__':main()

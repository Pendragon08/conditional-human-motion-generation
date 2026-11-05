"""Recheck rest asymmetry directly with zero-pose SMPL-X forward pass."""
import contextlib,hashlib,json,sys
from pathlib import Path
import numpy as np,torch,smplx
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MODELS=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
OUT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    torch.set_num_threads(1);r=json.loads((OUT/'report.json').read_text());assert sha(OUT/'run.py')==r['script_sha256']
    record=next(a for a in map(json.loads,(ROOT/'ssm_id_alignment_20261030/ids.jsonl').read_text().splitlines()) if a['motion_id']=='002511')
    path=Path(json.loads(Path(record['source_receipt']).read_text())['smplx_file']);assert sha(path)==r['source_sha256']
    with np.load(path,allow_pickle=False) as d:betas=np.asarray(d['betas'],dtype=np.float32).reshape(-1)[:16];gender=str(d['gender'].item())
    output=[]
    for row in r['results']:
        sex=row['gender'];assert sha(MODELS/'smplx'/('SMPLX_'+sex.upper()+'.npz'))==row['model_sha256']
        with contextlib.redirect_stdout(sys.stderr):model=smplx.create(str(MODELS),model_type='smplx',gender=sex,num_betas=16,use_pca=False,batch_size=1,ext='npz').cpu()
        beta=betas if sex==gender else np.zeros(16,dtype=np.float32);z=lambda n:torch.zeros((1,n),dtype=torch.float32)
        with torch.no_grad():j=model(betas=torch.tensor(beta[None,:]),global_orient=z(3),body_pose=z(63),transl=z(3),left_hand_pose=z(45),right_hand_pose=z(45),jaw_pose=z(3),leye_pose=z(3),reye_pose=z(3),expression=z(model.num_expression_coeffs),return_verts=False).joints[0,:22].numpy().astype(np.float64)
        j-=j[0]; reflected=j.copy();reflected[:,0]*=-1;left=[1,4,7,10,13,16,18,20];right=[2,5,8,11,14,17,19,21];temp=reflected[left].copy();reflected[left]=reflected[right];reflected[right]=temp
        distances=np.linalg.norm(j-reflected,axis=1);mean=float(distances.mean());maximum=float(distances.max())
        discrepancy=max(abs(mean-row['rest_joint_asymmetry_mean_m']),abs(maximum-row['rest_joint_asymmetry_max_m']));assert discrepancy<2e-6
        output.append({'gender':sex,'mean_difference_m':mean,'max_difference_m':maximum,'discrepancy_to_primary_m':discrepancy})
    result={'rows':output,'primary_report_sha256':sha(OUT/'report.json'),'audit_script_sha256':sha(Path(__file__)),'limitations':'Same existing model files; checks rest pose asymmetry only, not source timing, true human shape, full-body geometry, or training suitability.'}
    with (OUT/'independent_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()

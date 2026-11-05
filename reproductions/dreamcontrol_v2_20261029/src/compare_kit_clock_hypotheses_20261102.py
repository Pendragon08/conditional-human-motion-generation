"""Compare declared-120 and hypothetical-100 index maps on two KIT samples.

This is a sensitivity experiment, not proof of clock, identity, or license.
"""
import argparse,contextlib,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
import smplx,torch

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
KIT=ROOT/'kit_two_selected_20261102';REF=ROOT/'humanml_mirror_samples_20261102'
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
TRIPLES=((1,4,7),(2,5,8),(13,16,18),(14,17,19),(16,18,20),(17,19,21))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def angles(points):
    values=[]
    for a,b,c in TRIPLES:
        x=points[:,a]-points[:,b];y=points[:,c]-points[:,b]
        denom=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
        if np.any(denom<1e-8):raise ValueError('degenerate bone')
        values.append(np.sum(x*y,axis=1)/denom)
    return np.stack(values,axis=1)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(1)
    if shutil.disk_usage(ROOT).free<8*1024**3+64*1024**2:raise RuntimeError('disk budget')
    identity=json.loads((REF/'identity_comparison.json').read_text())
    if not identity['shapes_equal'] or identity['max_abs_diff']>1e-5:raise ValueError('mirror witness mismatch')
    a.out.mkdir(exist_ok=False);cases=[]
    for sid in ('004271','007555'):
        source=KIT/(sid+'_smplx_stageii.npz');reference=REF/(sid+'.npy');preview=KIT/('human_'+sid)/'human_20hz_declared.npz'
        report=json.loads((KIT/('human_'+sid)/'report.json').read_text());sh=sha(source);rh=sha(reference);ph=sha(preview)
        if sh!=report['source_sha256'] or ph!=report['artifact_sha256']:raise ValueError('source changed')
        with np.load(source,allow_pickle=False) as z:
            trans=z['trans'].copy();orient=z['root_orient'].copy();body=z['pose_body'].copy();betas=z['betas'][:16].astype(np.float32);gender=str(z['gender'].item());fps=float(z['mocap_frame_rate'])
        if fps!=120:raise ValueError('unexpected declared fps')
        model_path=MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz');model_hash=sha(model_path)
        with contextlib.redirect_stdout(sys.stderr):model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=16,use_pca=False,batch_size=1,ext='npz').cpu()
        idx=np.arange(0,len(trans),5);chunks=[]
        for s in range(0,len(idx),60):
            f=idx[s:s+60];n=len(f);t=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32);zero=lambda k:torch.zeros((n,k))
            with torch.no_grad():pts=model(betas=t(np.broadcast_to(betas,(n,16)).copy()),global_orient=t(orient[f]),body_pose=t(body[f]),transl=t(trans[f]),left_hand_pose=zero(45),right_hand_pose=zero(45),jaw_pose=zero(3),leye_pose=zero(3),reye_pose=zero(3),expression=zero(model.num_expression_coeffs),return_verts=False).joints[:,:22].numpy()
            chunks.append(pts)
        hypothetical=np.concatenate(chunks)
        with np.load(preview,allow_pickle=False) as z:declared=z['human'].copy();frames=z['source_frames'].copy()
        if not np.array_equal(frames,np.arange(0,len(trans),6)):raise ValueError('preview frame mismatch')
        common,left,right=np.intersect1d(idx,frames,return_indices=True)
        common_error=float(np.max(np.abs(hypothetical[left]-declared[right])))
        if common_error>1e-5:raise ValueError('common source frames do not match')
        target=np.load(reference,allow_pickle=False)
        if target.ndim!=3 or target.shape[1:]!=(22,3) or not np.isfinite(target).all():raise ValueError('bad mirror joints')
        count=min(len(declared),len(hypothetical),len(target));ref=angles(target[:count])
        if count<15:raise ValueError('too few paired samples')
        results={}
        for name,points in (('declared_120_stride6',declared),('hypothetical_100_stride5',hypothetical)):
            values=angles(points[:count]);diff=np.abs(values-ref)
            results[name]={'mean_abs_joint_cosine_difference':float(diff.mean()),'median_abs_joint_cosine_difference':float(np.median(diff)),
                           'per_joint_mean_abs_difference':[float(x) for x in diff.mean(axis=0)]}
        if any(sha(p)!=h for p,h in ((source,sh),(reference,rh),(preview,ph),(model_path,model_hash))):raise RuntimeError('input changed')
        cases.append({'id':sid,'source_sha256':sh,'mirror_sha256':rh,'existing_preview_sha256':ph,'model_sha256':model_hash,
                      'source_declared_fps':fps,'source_frames':len(trans),'mirror_joint_frames':len(target),'paired_prefix_frames':count,
                      'common_source_frames_checked':len(common),'common_frame_max_error_m':common_error,'comparisons':results})
    result={'cases':cases,'identity_comparison_sha256':sha(REF/'identity_comparison.json'),'script_sha256':sha(Path(__file__)),
            'joint_angle_cosine_triplets':[list(x) for x in TRIPLES],'training_ready':False,
            'limitations':['Third-party mirror is not the official raw recording; one other example was numerically close but not byte identical.',
             'Two clips only; joint-angle cosine errors are a heuristic, not a verified clock or source identity.',
             'Hypothetical 100 Hz is tested without changing the source declared 120 Hz or training outputs.',
             'Mirror processing, shape normalization, model and one-frame effects can affect joint angles.']}
    with (a.out/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()

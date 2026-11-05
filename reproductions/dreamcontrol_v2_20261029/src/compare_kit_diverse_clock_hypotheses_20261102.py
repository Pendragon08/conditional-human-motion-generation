"""Compare 5- vs 6-frame mappings on eight nominal KIT subject directories.

Third-party HumanML joints, SMPL-X model, and a joint-angle heuristic are not
raw-capture timestamps. The 5-frame hypothesis never changes source metadata.
"""
import argparse,contextlib,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np,smplx,torch
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'kit_diverse_clock_samples_20261102'
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
TRIPLES=((1,4,7),(2,5,8),(13,16,18),(14,17,19),(16,18,20),(17,19,21))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def angles(p):
    out=[]
    for a,b,c in TRIPLES:
        x=p[:,a]-p[:,b];y=p[:,c]-p[:,b];den=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
        if np.any(den<1e-8):raise ValueError('degenerate joints')
        out.append(np.sum(x*y,axis=1)/den)
    return np.stack(out,axis=1)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(1)
    if shutil.disk_usage(ROOT).free<8*1024**3+64*1024**2:raise RuntimeError('disk budget')
    src_report=BASE/'source_report.json';ref_report=BASE/'mirror_report.json';sh,rh=sha(src_report),sha(ref_report)
    sources=json.loads(src_report.read_text())['members'];refs=json.loads(ref_report.read_text())['found']
    if len(sources)!=8 or len({r['subject_nominal'] for r in sources})!=8:raise ValueError('unexpected source groups')
    a.out.mkdir(exist_ok=False);models={};cases=[]
    for row in sources:
        sid=row['id'];p=BASE/(sid+'_smplx_stageii.npz');q=BASE/(sid+'.npy')
        if sha(p)!=row['member_sha256'] or sha(q)!=refs[sid+'.npy']['sha256']:raise ValueError('source or mirror hash')
        with np.load(p,allow_pickle=False) as z:
            trans=z['trans'].copy();orient=z['root_orient'].copy();body=z['pose_body'].copy()
            betas=z['betas'][:16].astype(np.float32);gender=str(z['gender'].item());fps=float(z['mocap_frame_rate'])
        if fps!=120:raise ValueError('unexpected declared FPS')
        if gender not in models:
            with contextlib.redirect_stdout(sys.stderr):models[gender]=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=16,use_pca=False,batch_size=1,ext='npz').cpu()
        model=models[gender];model_path=MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz');model_hash=sha(model_path)
        raw5=np.arange(0,len(trans),5);raw6=np.arange(0,len(trans),6);idx=np.union1d(raw5,raw6);chunks=[]
        for start in range(0,len(idx),60):
            f=idx[start:start+60];n=len(f);t=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32);zero=lambda k:torch.zeros((n,k))
            with torch.no_grad():out=model(betas=t(np.broadcast_to(betas,(n,16)).copy()),global_orient=t(orient[f]),
                body_pose=t(body[f]),transl=t(trans[f]),left_hand_pose=zero(45),right_hand_pose=zero(45),jaw_pose=zero(3),
                leye_pose=zero(3),reye_pose=zero(3),expression=zero(model.num_expression_coeffs),return_verts=False).joints[:,:22].numpy()
            chunks.append(out)
        points=np.concatenate(chunks);p5=points[np.searchsorted(idx,raw5)];p6=points[np.searchsorted(idx,raw6)]
        reference=np.load(q,allow_pickle=False)
        if reference.ndim!=3 or reference.shape[1:]!=(22,3) or not np.isfinite(reference).all():raise ValueError('bad mirror joints')
        length=min(len(p5),len(p6),len(reference));reference_angles=angles(reference[:length]);results={}
        if length<15:raise ValueError('too few frames')
        for name,candidate in (('stride5',p5),('stride6',p6)):
            diff=np.abs(angles(candidate[:length])-reference_angles)
            results[name]={'mean_abs_joint_cosine_difference':float(diff.mean()),
                           'per_joint_mean_abs_difference':[float(x) for x in diff.mean(axis=0)]}
        if sha(p)!=row['member_sha256'] or sha(q)!=refs[sid+'.npy']['sha256'] or sha(model_path)!=model_hash:raise RuntimeError('changed input')
        cases.append({'id':sid,'source_member':row['member'],'source_sha256':row['member_sha256'],'mirror_sha256':refs[sid+'.npy']['sha256'],
            'nominal_subject':row['subject_nominal'],'model_sha256':model_hash,'declared_fps':fps,'source_frames':len(trans),
            'nominal_humanml_end':row['nominal_end'],'mirror_frames':len(reference),'compared_frames':length,'comparisons':results})
        print('CHECKED',sid,flush=True)
    if sha(src_report)!=sh or sha(ref_report)!=rh:raise RuntimeError('input report changed')
    better=sum(c['comparisons']['stride5']['mean_abs_joint_cosine_difference'] < c['comparisons']['stride6']['mean_abs_joint_cosine_difference'] for c in cases)
    result={'source_report_sha256':sh,'mirror_report_sha256':rh,'script_sha256':sha(Path(__file__)),'cases':cases,
        'case_count':len(cases),'nominal_subject_count':8,'stride5_lower_mean_error_cases':better,'training_ready':False,
        'limitations':['Eight nominal folders need not be eight people; recording identities unverified.',
          'Third-party mirror was not authenticated against raw data; the one unrelated official example was only numerically close.',
          'Model fitting and HumanML normalization can affect angle error; no paired raw C3D timing proof.',
          'Stride5 is a comparison hypothesis, not a replacement FPS or author preprocessing rule.']}
    with (a.out/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()

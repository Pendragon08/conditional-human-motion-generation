"""Build isolated 20 Hz KIT human previews under DART's 100 Hz hypothesis.

The source file retains its 120 Hz header. DART's dictionary is a third-party
name match, not confirmed capture time. No caption or sample is accepted.
"""
import contextlib,hashlib,json,re,shutil,sys,zipfile
from pathlib import Path
import numpy as np,smplx,torch
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'kit_diverse_clock_samples_20261102'
WORK=ROOT/'kit_dart_hypothesis_work_20261102'
OUT=ROOT/'kit_dart_hypothesis_human_20261102'
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
TEXTS=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')
DART_SHA='5842925e822df7e4a90d5c0e74dc0c0061ea442d0c3e9809042580e0a5d754f8'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def k(s):
    parts=s.replace('\\','/').split('/');name=re.sub(r'(?i)_(poses|stageii)\.(npy|npz)$','',parts[-1])
    return parts[-2],re.sub(r'[^a-z0-9]','',name.casefold())
def main():
    torch.set_num_threads(1)
    if shutil.disk_usage(ROOT).free<8*1024**3+128*1024**2:raise RuntimeError('disk budget')
    dart=WORK/'fps_dict_all.json';source_report=BASE/'source_report.json'; dh,sh,th=sha(dart),sha(source_report),sha(TEXTS)
    if dh!=DART_SHA:raise ValueError('DART dictionary changed')
    fps={}
    for name,value in json.loads(dart.read_text()).items():
        if name.startswith('KIT/'):
            key=k(name)
            if key in fps:raise ValueError('ambiguous DART name')
            fps[key]=(name,float(value))
    entries=json.loads(source_report.read_text())['members']
    if len(entries)!=8:raise ValueError('expected eight witnesses')
    OUT.mkdir(exist_ok=False); models={};cases=[]
    with zipfile.ZipFile(TEXTS) as captions:
      names=captions.namelist()
      if len(names)!=len(set(names)):raise ValueError('duplicate caption archive entry')
      for row in entries:
        sid=row['id'];source=BASE/(sid+'_smplx_stageii.npz')
        if sha(source)!=row['member_sha256']:raise ValueError('source hash mismatch')
        dname,rate=fps[k(row['member'])]
        if rate!=100.:raise ValueError('DART hypothesis is not 100 Hz')
        with np.load(source,allow_pickle=False) as z:
            original_fps=float(z['mocap_frame_rate']);gender=str(z['gender'].item());trans=z['trans'].copy();orient=z['root_orient'].copy()
            body=z['pose_body'].copy();betas=np.asarray(z['betas'][:16],np.float32)
        if original_fps!=120. or str(row['nominal_start'])!='0':raise ValueError('unexpected source metadata')
        model_path=MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz');model_sha=sha(model_path)
        if gender not in models:
            with contextlib.redirect_stdout(sys.stderr):models[gender]=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=16,use_pca=False,batch_size=1,ext='npz').cpu()
        model=models[gender];indices=np.arange(0,len(trans),5);chunks=[]
        for start in range(0,len(indices),60):
            ix=indices[start:start+60];n=len(ix);t=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32);zero=lambda m:torch.zeros((n,m))
            with torch.no_grad():pts=model(betas=t(np.broadcast_to(betas,(n,16)).copy()),global_orient=t(orient[ix]),body_pose=t(body[ix]),transl=t(trans[ix]),
                left_hand_pose=zero(45),right_hand_pose=zero(45),jaw_pose=zero(3),leye_pose=zero(3),reye_pose=zero(3),
                expression=zero(model.num_expression_coeffs),return_verts=False).joints[:,:22].numpy()
            chunks.append(pts)
        pts=np.concatenate(chunks);end=int(row['nominal_end'])
        if pts.shape!=(len(indices),22,3) or end>len(pts) or not np.isfinite(pts).all():raise ValueError('sample/bounds')
        capnames=[x for x in names if x in (sid+'.txt','texts/'+sid+'.txt')]
        if len(capnames)!=1:raise ValueError('caption archive member missing or ambiguous')
        raw=captions.read(capnames[0]);text=raw.decode('utf8');caption_sha=hashlib.sha256(raw).hexdigest()
        d=OUT/('human_'+sid);d.mkdir(exist_ok=False);target=d/'human_20hz_dart_hypothesis.npz'
        with target.open('xb') as f:np.savez_compressed(f,human=pts,source_frames=indices,source_times=indices/100.)
        result={'id':sid,'source_member':row['member'],'source_sha256':row['member_sha256'],'dart_key':dname,'dart_fps':rate,
          'source_declared_fps':original_fps,'assumed_fps_for_preview':100.,'hypothesis_only':True,
          'source_frames':len(trans),'output_points':len(indices),'nominal_index_interval':[0,end],
          'model_sha256':model_sha,'artifact_sha256':sha(target),'caption_archive_sha256':th,'caption_member':capnames[0],
          'caption_sha256':caption_sha,'caption_raw_utf8':text,'caption_status':'unreviewed_source_text',
          'training_ready':False,'limitations':['DART-based 100 Hz hypothesis, not measured clock; original header remains 120 Hz.',
            'Full source clip, not yet officially verified HumanML index identity or semantic text alignment.',
            'Direct every-five-frame sampling has no anti-alias filter.',
            'Locked-head SMPL-X with default hand/face, assumed ground and robot physics not checked.']}
        with (d/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
        if sha(source)!=row['member_sha256'] or sha(model_path)!=model_sha:raise RuntimeError('source or model changed')
        cases.append(result);print('HUMAN',sid,len(indices),flush=True)
    if sha(dart)!=dh or sha(source_report)!=sh or sha(TEXTS)!=th:raise RuntimeError('evidence changed')
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump({'cases':cases,'script_sha256':sha(Path(__file__)),
      'dart_dictionary_sha256':dh,'source_report_sha256':sh,'caption_archive_sha256':th,'training_ready':False},f,indent=2,allow_nan=False)
    print(json.dumps({'cases':len(cases),'points':sum(x['output_points'] for x in cases),'training_ready':False}),flush=True)
if __name__=='__main__':main()

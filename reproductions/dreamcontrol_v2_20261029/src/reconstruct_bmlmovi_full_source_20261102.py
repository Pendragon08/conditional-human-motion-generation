"""Reconstruct selected BMLmovi full-rate SMPL-X joint timelines.

Full source frames are retained. The header FPS and text association are
nominal, locked-head model and zero hands are approximations, no training grant.
"""
import contextlib
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import smplx
import torch

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'bmlmovi_diverse_source_20261102'
OUT=ROOT/'bmlmovi_full_human_20261102'
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
TEXTS=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def guard():
    if shutil.disk_usage(ROOT).free<8*1024**3+128*1024**2:raise RuntimeError('disk budget')


def main():
    torch.set_num_threads(1);guard()
    selection_path=BASE/'selection.json';report_path=BASE/'report.json';audit_path=BASE/'transfer_audit.json'
    selection_hash,report_hash,audit_hash,text_hash=map(sha,(selection_path,report_path,audit_path,TEXTS))
    selection=json.loads(selection_path.read_text(encoding='utf8'))
    report=json.loads(report_path.read_text(encoding='utf8'))
    audit=json.loads(audit_path.read_text(encoding='utf8'))
    if (selection['training_ready'] is not False or report['training_ready'] is not False or audit['training_ready'] is not False
        or report['selection_sha256']!=selection_hash or audit['selection_sha256']!=selection_hash
        or audit['extraction_report_sha256']!=report_hash):raise ValueError('source evidence mismatch')
    rows={r['id']:r for r in report['members']}; audited={r['id']:r for r in audit['cases']}
    if len(rows)!=20 or set(rows)!=set(audited) or set(rows)!={r['id'] for r in selection['members']}:
        raise ValueError('source selection coverage mismatch')
    OUT.mkdir(exist_ok=False);models={};failures=0
    with zipfile.ZipFile(TEXTS) as captions,(OUT/'records.jsonl').open('x',encoding='utf8',buffering=1) as log:
      names=captions.namelist()
      if len(names)!=len(set(names)):raise ValueError('duplicate caption archive entry')
      for selected in selection['members']:
        guard();sid=selected['id'];row=rows[sid];source=BASE/row['output']
        result={'id':sid,'source_member':row['member'],'source_sha256':row['member_sha256'],'training_ready':False}
        try:
            if sha(source)!=row['member_sha256'] or sha(source)!=audited[sid]['sha256']:
                raise ValueError('source hash mismatch')
            with np.load(source,allow_pickle=False) as z:
                fps=float(z['mocap_frame_rate']);gender=str(z['gender'].item());trans=z['trans'].copy()
                orient=z['root_orient'].copy();body=z['pose_body'].copy();betas=np.asarray(z['betas'][:16],np.float32)
            n=len(trans)
            if (fps!=120. or n<2 or n>1200 or trans.shape!=(n,3) or orient.shape!=(n,3)
                or body.shape[0]!=n or not np.isfinite(trans).all() or not np.isfinite(orient).all() or not np.isfinite(body).all()):
                raise ValueError('source timeline/parameters invalid')
            model_path=MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz');model_sha=sha(model_path)
            if gender not in models:
                with contextlib.redirect_stdout(sys.stderr):
                    models[gender]=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=16,use_pca=False,batch_size=1,ext='npz').cpu()
            model=models[gender];chunks=[]
            for start in range(0,n,60):
                guard();end=min(start+60,n);size=end-start
                t=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32)
                zero=lambda width:torch.zeros((size,width))
                with torch.no_grad():
                    points=model(betas=t(np.broadcast_to(betas,(size,16)).copy()),global_orient=t(orient[start:end]),
                        body_pose=t(body[start:end]),transl=t(trans[start:end]),left_hand_pose=zero(45),right_hand_pose=zero(45),
                        jaw_pose=zero(3),leye_pose=zero(3),reye_pose=zero(3),expression=zero(model.num_expression_coeffs),
                        return_verts=False).joints[:,:22].numpy()
                chunks.append(points)
            human=np.concatenate(chunks)
            if human.shape!=(n,22,3) or not np.isfinite(human).all():raise ValueError('invalid output points')
            matches=[x for x in names if x in (sid+'.txt','texts/'+sid+'.txt')]
            if len(matches)!=1:raise ValueError('caption archive member missing or ambiguous')
            raw=captions.read(matches[0]);raw.decode('utf8')
            dest=OUT/(sid+'.npz')
            with dest.open('xb') as f:
                np.savez_compressed(f,human_joints=human,source_frames=np.arange(n,dtype=np.int64),declared_fps=np.array(fps))
            result.update({'status':'reconstructed_unreviewed','artifact':dest.name,'artifact_sha256':sha(dest),
                'frames':n,'declared_fps':fps,'nominal_index_interval':selected['nominal_index_interval'],
                'official_split':selected['official_split'],'auxiliary_split':selected['auxiliary_split'],
                'model_sha256':model_sha,'caption_member':matches[0],'caption_sha256':hashlib.sha256(raw).hexdigest(),
                'caption_archive_sha256':text_hash,'caption_accepted':False})
            if sha(source)!=row['member_sha256'] or sha(model_path)!=model_sha:raise RuntimeError('source/model changed')
            failures=0
        except Exception as exc:
            result.update({'status':'failed','error':f'{type(exc).__name__}: {exc}'});failures+=1
        log.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps(result,ensure_ascii=False),flush=True)
        if failures>=3:raise RuntimeError('three consecutive sample failures')
    if (sha(selection_path)!=selection_hash or sha(report_path)!=report_hash or sha(audit_path)!=audit_hash
        or sha(TEXTS)!=text_hash):raise RuntimeError('input changed')
    summary={'selection_sha256':selection_hash,'source_report_sha256':report_hash,'transfer_audit_sha256':audit_hash,
             'caption_archive_sha256':text_hash,'records_sha256':sha(OUT/'records.jsonl'),
             'script_sha256':sha(Path(__file__)),'requested':len(rows),'training_ready':False,
             'limitations':['Full source rate with declared, unverified FPS and no robot retargeting.',
                 'Locked-head SMPL-X and zero hand/face poses are approximations.',
                 'Name, text, timing, physical feasibility and license remain unaccepted.']}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(summary,f,indent=2,allow_nan=False)


if __name__=='__main__':main()

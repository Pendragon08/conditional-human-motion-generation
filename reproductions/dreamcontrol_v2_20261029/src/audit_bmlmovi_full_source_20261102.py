"""Re-read selected BMLmovi outputs and recompute three source frames each."""
import contextlib
import hashlib
import json
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
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    report_path=OUT/'report.json';report=json.loads(report_path.read_text())
    selected=json.loads((BASE/'selection.json').read_text(encoding='utf8'))['members']
    source_rows={r['id']:r for r in json.loads((BASE/'report.json').read_text(encoding='utf8'))['members']}
    rows=[json.loads(x) for x in (OUT/'records.jsonl').read_text(encoding='utf8').splitlines()]
    if (sha(OUT/'records.jsonl')!=report['records_sha256'] or sha(BASE/'selection.json')!=report['selection_sha256']
        or sha(BASE/'report.json')!=report['source_report_sha256'] or sha(TEXTS)!=report['caption_archive_sha256']
        or len(rows)!=20 or [x['id'] for x in rows]!=[x['id'] for x in selected]):
        raise ValueError('source or selection binding mismatch')
    cases=[]
    with zipfile.ZipFile(TEXTS) as captions:
      for row in rows:
        if row['status']!='reconstructed_unreviewed':
            cases.append({'id':row['id'],'status':'source_batch_failed','error':row.get('error')});continue
        sid=row['id'];source=BASE/source_rows[sid]['output'];artifact=OUT/row['artifact']
        if (sha(source)!=row['source_sha256'] or sha(artifact)!=row['artifact_sha256']
            or hashlib.sha256(captions.read(row['caption_member'])).hexdigest()!=row['caption_sha256']):
            raise ValueError('source/feature/caption hash mismatch')
        with np.load(source,allow_pickle=False) as z:
            trans=z['trans'].copy();body=z['pose_body'].copy();orient=z['root_orient'].copy()
            betas=z['betas'][:16].astype(np.float32);gender=str(z['gender'].item());fps=float(z['mocap_frame_rate'])
        with np.load(artifact,allow_pickle=False) as z:
            points=z['human_joints'].copy();indices=z['source_frames'].copy();stored_fps=float(z['declared_fps'])
        model_file=MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz')
        if (sha(model_file)!=row['model_sha256'] or fps!=stored_fps or fps!=row['declared_fps']
            or points.shape!=(len(trans),22,3) or not np.isfinite(points).all()
            or not np.array_equal(indices,np.arange(len(trans)))):
            raise ValueError('model or full timeline mismatch')
        with contextlib.redirect_stdout(sys.stderr):
            model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=16,use_pca=False,batch_size=1,ext='npz').cpu()
        samples=[0,len(indices)//2,len(indices)-1];error=0.
        with torch.no_grad():
          for j in samples:
            zero=lambda n:torch.zeros((1,n),dtype=torch.float32)
            t=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32)
            value=model(betas=t(betas[None]),global_orient=t(orient[j][None]),body_pose=t(body[j][None]),
                transl=t(trans[j][None]),left_hand_pose=zero(45),right_hand_pose=zero(45),jaw_pose=zero(3),
                leye_pose=zero(3),reye_pose=zero(3),expression=zero(model.num_expression_coeffs),
                return_verts=False).joints[0,:22].numpy()
            error=max(error,float(np.max(np.abs(value-points[j]))))
        if not np.isfinite(error) or error>1e-5:raise ValueError('human sampled reconstruction mismatch')
        cases.append({'id':sid,'status':'technical_sample_checked','artifact_sha256':sha(artifact),
            'source_sha256':sha(source),'frames':len(indices),'checked_indices':samples,
            'max_abs_difference_m':error,'caption_sha256':row['caption_sha256']})
    result={'cases':cases,'source_report_sha256':sha(report_path),'script_sha256':sha(Path(__file__)),
            'training_ready':False,'limitations':['Only three frames recomputed using the same SMPL-X model.',
              'Caption hashes do not establish semantic match; declared FPS is not verified time.',
              'No robot retargeting, contact, physics or training acceptance.']}
    with (OUT/'audit.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'cases':len(cases),'checked':sum(c['status']=='technical_sample_checked' for c in cases),
                      'failed':sum(c['status']!='technical_sample_checked' for c in cases),'training_ready':False}))


if __name__=='__main__':main()

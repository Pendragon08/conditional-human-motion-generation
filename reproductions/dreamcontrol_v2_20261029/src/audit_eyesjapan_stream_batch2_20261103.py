"""Audit stream output and independently rerun stored source witnesses with same model."""
import contextlib, hashlib, json, sys
from pathlib import Path
import numpy as np
import smplx
import torch
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 mode=sys.argv[1]
 if mode != 'batch2':raise ValueError('invalid mode')
 queue=ROOT/('eyesjapan_stream_'+mode+'_queue_20261103.json');out=ROOT/('eyesjapan_stream_'+mode+'_20261103')
 q=json.loads(queue.read_text()); report=json.loads((out/'report.json').read_text());rows=[json.loads(x) for x in (out/'records.jsonl').read_text().splitlines()]
 expected={r['member']:r for r in q['members']}
 if report['queue_sha256']!=sha(queue) or report['records_sha256']!=sha(out/'records.jsonl') or len(rows)!=len(expected) or {r['member'] for r in rows}!=set(expected):raise ValueError('coverage/hash mismatch')
 torch.set_num_threads(1);models={};cases=[]
 for row in rows:
  try:
   ref=expected[row['member']];p=out/row['artifact'];mp=MODEL/'smplx'/('SMPLX_'+str(np.load(p,allow_pickle=False)['witness_gender'].item()).upper()+'.npz')
   if row['status']!='reconstructed_unreviewed' or row['source_sha256']!=ref['source_sha256'] or sha(p)!=row['artifact_sha256'] or sha(mp)!=row['model_sha256']:raise ValueError('source/artifact/model hash')
   with np.load(p,allow_pickle=False) as z:
    human=z['human_joints'];frames=z['source_frames'];idx=z['witness_frames'];fps=float(z['declared_fps']);gender=str(z['witness_gender'].item());betas=z['witness_betas'];trans=z['witness_trans'];orient=z['witness_orient'];body=z['witness_body']
   n=ref['source_frames']
   if human.shape!=(n,22,3) or not np.isfinite(human).all() or not np.array_equal(frames,np.arange(n)) or not np.array_equal(idx,[0,n//2,n-1]) or fps!=ref['declared_fps']:raise ValueError('frame coverage')
   if gender not in models:
    with contextlib.redirect_stdout(sys.stderr): models[gender]=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=16,use_pca=False,batch_size=1,ext='npz').cpu()
   model=models[gender]; t=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32);zero=lambda k:torch.zeros((3,k))
   with torch.no_grad():
    actual=model(betas=t(np.broadcast_to(betas,(3,16)).copy()),global_orient=t(orient),body_pose=t(body),transl=t(trans),left_hand_pose=zero(45),right_hand_pose=zero(45),jaw_pose=zero(3),leye_pose=zero(3),reye_pose=zero(3),expression=zero(model.num_expression_coeffs),return_verts=False).joints[:,:22].numpy()
   diff=float(np.max(np.abs(actual-human[idx])))
   if diff>2e-6:raise ValueError('witness difference')
   cases.append({'member':row['member'],'status':'technical_sample_checked','source_sha256':row['source_sha256'],'artifact_sha256':row['artifact_sha256'],'frames':n,'max_sampled_difference_m':diff})
  except Exception as e:cases.append({'member':row['member'],'status':'failed','error':f'{type(e).__name__}: {e}'})
 audit={'queue_sha256':sha(queue),'report_sha256':sha(out/'report.json'),'records_sha256':sha(out/'records.jsonl'),'cases':cases,'training_ready':False,'limitations':['Same model and stored witnesses, not source-clock, semantic, physical or training validation.']}
 with (out/'audit.json').open('x',encoding='utf8') as f:json.dump(audit,f,indent=2,allow_nan=False)
 print(json.dumps({'cases':len(cases),'checked':sum(c['status']=='technical_sample_checked' for c in cases),'failed':sum(c['status']!='technical_sample_checked' for c in cases)}))
if __name__=='__main__':main()

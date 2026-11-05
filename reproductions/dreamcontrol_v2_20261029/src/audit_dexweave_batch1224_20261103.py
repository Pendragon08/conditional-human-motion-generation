"""Read back completed GRAB technical preview, against pinned queue and captions."""
import hashlib,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'dexweave_stream_robot_batch1224_20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
def main():
 qp=R/'grab_dex_batch1224_queue_20261103.jsonl';cp=R/'grab_dex_candidate_pairs_20261103.jsonl'
 if sha(qp)!='eadb6ad1a8c49baf13a0702e22e284348b60de2cca4a02159b8d43261de180b0':raise ValueError('queue hash')
 if sha(cp)!='ba68debdf5061aaf7f3b5fa3cf8ab64c4b1266f439bca187cbbce5b13f849104':raise ValueError('candidates hash')
 q=load(qp);c=load(cp);a=load(D/'records.jsonl');ix={x['id']:x for x in c}
 if len(q)!=1224 or len(a)!=len(q) or len(ix)!=1335 or len({x['id'] for x in q})!=len(q):raise ValueError('coverage')
 report=json.loads((D/'report.json').read_text())
 if report['count']!=len(q):raise ValueError('producer count')
 details=[];errors=[];frames=0
 for index,(x,y) in enumerate(zip(q,a)):
  try:
   if x['id']!=y['id'] or y['id'] not in ix:raise ValueError('order or identity')
   z=ix[y['id']]
   for field in ('source_member_sha256','text_sha256'):
    if y[field]!=x[field] or y[field]!=z[field]:raise ValueError(field)
   if y['captions']!=z['captions'] or y['risk_codes']!=z['risk_codes']:raise ValueError('caption binding')
   if y['caption_accepted'] or y['training_ready'] or y['status']!='technical_preview':raise ValueError('preview status')
   path=D/y['artifact']
   if sha(path)!=y['artifact_sha256']:raise ValueError('output hash')
   with np.load(path,allow_pickle=False) as f:
    state=np.asarray(f['state29_wxyz_xyz']);x263=np.asarray(f['features']);pts=np.asarray(f['points_z_up']);tm=np.asarray(f['sample_times']);names=f['joint_names'];obj=np.asarray(f['object_pose_wxyz_xyz'])
   n=len(state)
   if state.shape!=(n,36) or x263.shape!=(n-1,263) or pts.shape!=(n,22,3) or tm.shape!=(n,) or len(names)!=29 or len(set(names.tolist()))!=29 or obj.shape[0]!=n or obj.shape[-1]!=7:raise ValueError('shape')
   if not all(np.isfinite(v).all() for v in (state,x263,pts,tm,obj)):raise ValueError('finite')
   if not np.allclose(tm,np.arange(n)/20.,atol=1e-12,rtol=0) or not np.allclose(np.linalg.norm(state[:,:4],axis=1),1.,atol=1e-5):raise ValueError('time or quaternion')
   if n!=y['states'] or len(x263)!=y['feature_frames'] or y['source_fps']!=50.:raise ValueError('metadata')
   frames+=len(x263);details.append({'id':y['id'],'status':'passed_technical_reread','artifact_sha256':y['artifact_sha256'],'feature_frames':len(x263)})
  except Exception as e:errors.append({'index':index,'id':y.get('id'),'error':str(e)})
 result={'samples':len(a),'passed':len(details),'errors':errors,'feature_frames':frames,'records_sha256':sha(D/'records.jsonl'),
         'queue_sha256':sha(qp),'candidate_manifest_sha256':sha(cp),'details':details,'training_ready':False,
         'limitations':['Caption binding is nominal, not semantic acceptance.','Source time is nominal and robot model compatibility not established.','Not a physical-action filter.']}
 with (D/'audit.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,allow_nan=False)
 print(json.dumps({k:v for k,v in result.items() if k!='details'}))
if __name__=='__main__':main()

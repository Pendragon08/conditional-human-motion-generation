"""Independent full-record provenance, shape and time audit for a batch."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args()
 if not a.folder.replace('_','').isalnum():raise ValueError('invalid folder')
 folder=R/a.folder;report=json.loads((folder/'report.json').read_text());records=folder/'records.jsonl'
 if sha(records)!=report['records_sha256'] or any(sha(Path(p))!=v for p,v in report['input_sha256'].items()):raise ValueError('report input hash mismatch')
 rows=[json.loads(x) for x in records.read_text(encoding='utf8').splitlines()]
 if len(rows)!=report['count'] or [x['index'] for x in rows]!=list(range(report['start'],report['start']+report['count'])):raise ValueError('coverage')
 source=R/'data/OmniRetarget_Dataset/robot-object.zip';ann=R/'omomo_text_pairing_20261101/omomo_text_anno.zip';fail=[];passed=[]
 with zipfile.ZipFile(source) as z,zipfile.ZipFile(ann) as tz:
  for r in rows:
   if r['status']!='technical_preview':fail.append({'index':r['index'],'reason':'source preview failed'});continue
   raw=z.read(r['source_member']);annotation=tz.read(r['annotation_member'])
   if hashlib.sha256(raw).hexdigest()!=r['source_sha256'] or hashlib.sha256(annotation).hexdigest()!=r['annotation_sha256']:raise ValueError('member hash mismatch')
   if json.loads(annotation)[r['nominal_action_key']]!=r['caption'] or r['caption_accepted'] is not False:raise ValueError('caption binding or false acceptance')
   target=folder/r['artifact']
   if target.parent!=folder or sha(target)!=r['artifact_sha256']:raise ValueError('artifact mismatch')
   with np.load(io.BytesIO(raw),allow_pickle=False) as x:q0=x['qpos'];fps=float(x['fps'])
   with np.load(target,allow_pickle=False) as x:q=x['qpos29_source_layout_20hz'];t=x['sample_times'];feat=x['features'];points=x['points_z_up'];names=x['joint_names']
   n=len(q)
   if q.shape!=(n,43) or n<2 or n>200 or len(q0)!=r['source_frames'] or fps!=r['fps_declared'] or fps!=30 or len(names)!=29 or feat.shape!=(n-1,263) or points.shape!=(n,22,3):raise ValueError('shape')
   if not all(np.isfinite(x).all() for x in (q,feat,points,t)) or not np.allclose(t,np.arange(n)/20.,atol=1e-12,rtol=0) or t[-1]>(len(q0)-1)/fps+1e-9:raise ValueError('time or finite')
   if not np.allclose(np.linalg.norm(q[:,:4],axis=1),1,atol=1e-6,rtol=0) or not np.allclose(np.linalg.norm(q[:,36:40],axis=1),1,atol=1e-6,rtol=0):raise ValueError('quaternion')
   passed.append({'index':r['index'],'artifact_sha256':sha(target),'feature_frames':len(feat),'cropped':r['cropped_for_max_10s'],'caption_accepted':False})
 result={'records_sha256':sha(records),'cases':passed,'technical_failures':fail,'accepted_captions':0,'training_ready':False,'limitations':['Does not independently recalculate full FIR, FK or 263D encoding.','Does not establish caption semantics, physical timing, source rights or actual identity.','No physical-action filtering.']}
 with (folder/'audit.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps({'passed':len(passed),'failed':len(fail),'accepted_captions':0}))
if __name__=='__main__':main()

"""Independently verify partition coverage, provenance, and weak-stage training statistics."""
import hashlib,json,sys
from pathlib import Path
from collections import Counter
import numpy as np,torch
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'grab_omni_weaklabel_stage_20261103';F=R/'grab_omni_mixed29_full_preview_20261103'
sys.path.insert(0,'/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen')
from data_loaders.humanml.scripts.motion_process import recover_from_ric
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);rep=json.loads((S/'report.json').read_text())
 for name,key in [('manifest.json','manifest_sha256'),('priority_review.json','priority_review_sha256'),('stats.npz','stats_sha256')]:
  if sha(S/name)!=rep[key]:raise ValueError('hash mismatch')
 if sha(F/'manifest.json')!=rep['full_manifest_sha256']:raise ValueError('source manifest hash')
 source={x['sample_id']:x for x in json.loads((F/'manifest.json').read_text())['records']}
 stage=json.loads((S/'manifest.json').read_text())['records'];held=json.loads((S/'priority_review.json').read_text())['records']
 ids=[r['sample_id'] for r in stage+held]
 if len(ids)!=3161 or len(set(ids))!=3161 or set(ids)!=set(source):raise ValueError('partition mismatch')
 for x in stage+held:
  orig=source[x['sample_id']]
  if any(x[k]!=orig[k] for k in ('state_sha256','feature_sha256','caption','annotation_sha256','split','source_sha256')):raise ValueError('provenance mismatch')
  if x['caption_accepted'] or x['training_ready'] or x['label_assurance']!='source_paired_weak_unreviewed':raise ValueError('acceptance mismatch')
 if any(x['triage_reasons'] for x in stage) or any(not x['triage_reasons'] for x in held):raise ValueError('priority partition')
 sf=np.zeros(263);sq=np.zeros(263);sp=np.zeros(66);spq=np.zeros(66);n=0;counts=Counter()
 for x in stage:
  p=R/x['feature_path']
  if sha(p)!=x['feature_sha256']:raise ValueError('artifact hash')
  with np.load(p,allow_pickle=False) as f:z=np.asarray(f['features'],dtype=np.float32)
  counts[x['dataset_source']+':'+x['split']]+=1
  if x['split']=='train':
   with torch.no_grad():v=recover_from_ric(torch.from_numpy(z),22).numpy().reshape(-1,66).astype(np.float64)
   a=z.astype(np.float64);n+=len(a);sf+=a.sum(axis=0);sq+=(a*a).sum(axis=0);sp+=v.sum(axis=0);spq+=(v*v).sum(axis=0)
 if dict(counts)!=rep['counts'] or n!=rep['train_frames']:raise ValueError('count mismatch')
 expected={'feature_mean':sf/n,'feature_std':np.maximum(np.sqrt(np.maximum(sq/n-(sf/n)**2,0)),.001),
           'spatial_mean':sp/n,'spatial_std':np.maximum(np.sqrt(np.maximum(spq/n-(sp/n)**2,0)),.001)}
 with np.load(S/'stats.npz',allow_pickle=False) as x:errors={k:float(np.max(np.abs(x[k]-v))) for k,v in expected.items()}
 if max(errors.values())>1e-6:raise ValueError('stats mismatch')
 out={'source_samples':len(source),'staged':len(stage),'priority_review':len(held),'train_frames':n,
      'stats_max_abs_errors':errors,'manifest_sha256':rep['manifest_sha256'],'priority_sha256':rep['priority_review_sha256'],
      'training_ready':False,'caption_accepted':0,'limits':['Complete technical partition; no caption semantic or license acceptance.']}
 with (S/'independent_audit.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
 print(json.dumps(out))
if __name__=='__main__':main()

"""Independent readback and train-statistics check for the mixed technical preview."""
import hashlib,json,sys
from pathlib import Path
from collections import Counter
import numpy as np,torch
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'grab_omni_mixed29_baseline_preview_20261103'
T=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen')
sys.path.insert(0,str(T))
from data_loaders.humanml.scripts.motion_process import recover_from_ric
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);report=json.loads((D/'report.json').read_text())
 if sha(D/'manifest.json')!=report['manifest_sha256'] or sha(D/'stats.npz')!=report['stats_sha256']:raise ValueError('manifest or stats hash')
 rows=json.loads((D/'manifest.json').read_text())['records']; ids=set(); subjects={};source_hashes={};counts=Counter();n=0
 sf=np.zeros(263);sf2=np.zeros(263);sp=np.zeros(66);sp2=np.zeros(66)
 for r in rows:
  if r['sample_id'] in ids or r['caption_accepted'] or r['training_ready'] or r['fps']!=20.:raise ValueError('record flags')
  ids.add(r['sample_id']);subject=r['group_id']; previous=subjects.setdefault(subject,r['split'])
  if previous!=r['split']:raise ValueError('subject split leakage')
  source=r['source_sha256'];previous=source_hashes.setdefault(source,r['split'])
  if previous!=r['split']:raise ValueError('source hash split leakage')
  path=R/r['feature_path']
  if sha(path)!=r['feature_sha256']:raise ValueError('artifact checksum')
  with np.load(path,allow_pickle=False) as f:x=np.asarray(f['features'],dtype=np.float32)
  counts[r['dataset_source']+':'+r['split']]+=1
  if r['split']=='train':
   with torch.no_grad():y=recover_from_ric(torch.from_numpy(x),22).numpy().reshape(-1,66).astype(np.float64)
   z=x.astype(np.float64);n+=len(x);sf+=z.sum(axis=0);sf2+=(z*z).sum(axis=0);sp+=y.sum(axis=0);sp2+=(y*y).sum(axis=0)
 if dict(counts)!=report['samples_by_source_split'] or n!=report['train_frames']:raise ValueError('sample or frame count')
 with np.load(D/'stats.npz',allow_pickle=False) as st:
  expected={'feature_mean':sf/n,'feature_std':np.maximum(np.sqrt(np.maximum(sf2/n-(sf/n)**2,0)),.001),
            'spatial_mean':sp/n,'spatial_std':np.maximum(np.sqrt(np.maximum(sp2/n-(sp/n)**2,0)),.001)}
  errors={k:float(np.max(np.abs(st[k]-v))) for k,v in expected.items()}
 if max(errors.values())>1e-6:raise ValueError('stats mismatch '+str(errors))
 result={'manifest_sha256':report['manifest_sha256'],'stats_sha256':report['stats_sha256'],'samples':len(rows),
         'train_frames':n,'max_stat_abs_errors':errors,'source_split_counts':dict(counts),
         'nominal_subjects_checked':len(subjects),'source_hashes_checked':len(source_hashes),
         'caption_accepted':0,'training_ready':False,'physical_action_filter':False}
 with (D/'independent_audit.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps(result))
if __name__=='__main__':main()

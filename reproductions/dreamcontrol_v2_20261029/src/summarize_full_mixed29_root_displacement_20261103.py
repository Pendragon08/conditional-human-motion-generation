"""Summarize 20 Hz baseline reference trajectories; root motion is not a task label."""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'grab_omni_mixed29_full_preview_20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 report=json.loads((D/'report.json').read_text());p=D/'manifest.json'
 if sha(p)!=report['manifest_sha256']:raise ValueError('manifest changed')
 rows=json.loads(p.read_text())['records']; bins=defaultdict(Counter); examples=defaultdict(list)
 for r in rows:
  path=R/r['state_path']
  if sha(path)!=r['state_sha256']:raise ValueError('artifact hash')
  with np.load(path,allow_pickle=False) as f:
   key=next(k for k in ('state29_wxyz_xyz','qpos29_source_layout_20hz') if k in f)
   state=np.asarray(f[key],dtype=np.float64)
  if state.ndim!=2 or state.shape[1]<36 or not np.isfinite(state).all():raise ValueError('bad state')
  xy=state[:,4:6];displacement=float(np.max(np.linalg.norm(xy-xy[0],axis=1)))
  distance=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())
  band='lt_0_10m' if displacement<.1 else '0_10_to_lt_0_50m' if displacement<.5 else 'ge_0_50m'
  source=r['dataset_source'];bins[source][band]+=1
  tag=source+':'+band
  if len(examples[tag])<2:examples[tag].append({'sample_id':r['sample_id'],'caption':r['caption'],'max_displacement_m':round(displacement,5),'root_xy_path_m':round(distance,5)})
 result={'manifest_sha256':report['manifest_sha256'],'source_counts':{k:dict(v) for k,v in bins.items()},'examples':dict(examples),
         'measure':'maximum horizontal distance of the robot root from the first state, meters, over 20Hz preview',
         'limitations':['Thresholds are descriptive only, not a locomotion or manipulation classifier.',
                        'Stepping in place and rotation can occur with little root translation.',
                        'Root translation is not evidence of hand/object contact, task semantics or quality.',
                        'Reference clips can be cropped and Omni augmented tracks are not necessarily independent.']}
 with (D/'root_displacement_summary.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

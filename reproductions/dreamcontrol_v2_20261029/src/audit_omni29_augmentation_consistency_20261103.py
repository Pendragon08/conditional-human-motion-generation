"""Compare nominal Omni original/variant motion without using a semantic proxy as approval.

Checks source binding, frame counts, 29-joint differences and object origin in
the robot-root frame. This tests a rigid-global-transform hypothesis; mismatch
is diagnostic, not a physical-action or caption-semantic filter.
"""
import hashlib,json,shutil
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'grab_omni_mixed29_full_preview_20261103'
O=R/'omni29_augmentation_consistency_20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(r):
 p=R/r['state_path']
 if sha(p)!=r['state_sha256']:raise ValueError('artifact hash')
 with np.load(p,allow_pickle=False) as f:x=np.asarray(f['qpos29_source_layout_20hz'],dtype=np.float64)
 if x.ndim!=2 or x.shape[1]!=43 or not np.isfinite(x).all():raise ValueError('qpos shape')
 q=Rotation.from_quat(x[:,[1,2,3,0]])
 rel=q.inv().apply(x[:,40:43]-x[:,4:7])
 return x,rel
def main():
 if O.exists():raise FileExistsError(O)
 if shutil.disk_usage(R).free<8*1024**3+356*1024**2:raise RuntimeError('disk reserve')
 p=D/'manifest.json';report=json.loads((D/'report.json').read_text())
 if sha(p)!=report['manifest_sha256']:raise ValueError('manifest hash')
 groups=defaultdict(list)
 for row in json.loads(p.read_text())['records']:
  if row['dataset_source']=='OmniRetarget/OMOMO':groups[row['source_id']].append(row)
 result=[];errors=[];status=Counter()
 for group,rows in groups.items():
  originals=[x for x in rows if x['source_member'].endswith('_original.npz')]
  if len(originals)!=1:raise ValueError('original coverage')
  base,brel=read(originals[0])
  for v in rows:
   if v is originals[0]:continue
   curr,crel=read(v)
   if len(curr)!=len(base):
    status['length_mismatch']+=1;result.append({'id':v['sample_id'],'group':group,'status':'length_mismatch','original_frames':len(base),'variant_frames':len(curr)});continue
   delta=np.abs(curr[:,7:36]-base[:,7:36]);obj=np.linalg.norm(crel-brel,axis=1)
   item={'id':v['sample_id'],'group':group,'original_id':originals[0]['sample_id'],'variant_member':v['source_member'],
         'frames':len(curr),'joint_max_abs_rad':float(delta.max()),'joint_mean_abs_rad':float(delta.mean()),
         'object_root_max_m':float(obj.max()),'object_root_mean_m':float(obj.mean()),
         'both_clip_10s':bool(v['cropped_for_max_10s'] and originals[0]['cropped_for_max_10s']),
         'status':'compared_not_semantic_acceptance'}
   status['compared']+=1;result.append(item)
 if len(result)!=1514:raise ValueError('variant count')
 good=[x for x in result if x['status']=='compared_not_semantic_acceptance']
 summary={'manifest_sha256':sha(p),'nominal_action_groups':len(groups),'variant_count':len(result),'status_counts':dict(status),
   'joint_max_rad_quantiles':{str(q):float(np.quantile([x['joint_max_abs_rad'] for x in good],q)) for q in (0,.5,.9,1)},
   'object_root_max_m_quantiles':{str(q):float(np.quantile([x['object_root_max_m'] for x in good],q)) for q in (0,.5,.9,1)},
   'counts_joint_max_above_0_001rad':sum(x['joint_max_abs_rad']>.001 for x in good),
   'counts_object_root_max_above_0_001m':sum(x['object_root_max_m']>.001 for x in good),
   'limit':'Numerical differences test a rigid-global-transform hypothesis only; they neither prove caption correctness nor measure contact, tracking or physical feasibility.',
   'training_ready':False}
 O.mkdir(exist_ok=False)
 with (O/'records.jsonl').open('x',encoding='utf8') as f:
  for row in result:f.write(json.dumps(row,allow_nan=False)+'\n')
 summary['records_sha256']=sha(O/'records.jsonl')
 with (O/'report.json').open('x') as f:json.dump(summary,f,indent=2,allow_nan=False)
 print(json.dumps(summary))
if __name__=='__main__':main()

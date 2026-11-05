"""Read back the packaged baseline against pinned sources and exercise its loader."""
import hashlib,json,sys
from collections import Counter
from pathlib import Path
import numpy as np,torch
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
O=R/'grab_omni_29dof_weak_baseline_v1_20261103';F=R/'grab_omni_mixed29_full_preview_20261103';T=R/'grab_omni_caption_time_triage_20261103'
sys.path[:0]=[str(R/'robot_loader_20261030/src'),'/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen']
from robot_motion_dataset import RobotMotionDataset,collate_robot_motion
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);rep=json.loads((O/'report.json').read_text());mp=O/'manifest.json';sp=O/'stats.npz'
 if sha(mp)!=rep['manifest_sha256'] or sha(sp)!=rep['stats_sha256'] or sha(F/'manifest.json')!=rep['source_manifest_sha256'] or sha(T/'all_candidates.jsonl')!=rep['source_triage_sha256']:raise ValueError('package/source hash')
 full=json.loads((F/'manifest.json').read_text())['records'];packed=json.loads(mp.read_text())['records'];ti=[json.loads(x) for x in (T/'all_candidates.jsonl').read_text().splitlines()]
 old={x['sample_id']:x for x in full};tri={x['sample_id']:x for x in ti}
 if len(old)!=3161 or len(tri)!=3161 or len(packed)!=3161 or len({x['sample_id'] for x in packed})!=3161:raise ValueError('coverage')
 counts=Counter();reasons=Counter();crops=0
 for r in packed:
  a=old[r['sample_id']];b=tri[r['sample_id']]
  for k in ('caption','caption_source','state_path','state_sha256','feature_path','feature_sha256','source_sha256','annotation_sha256','split','fps','cropped_for_max_10s'):
   if r[k]!=a[k]:raise ValueError('source mismatch '+k)
  if r['caption_risk_reasons']!=b['triage_reasons'] or r['label_semantics_verified'] or r['caption_accepted'] or r['training_ready']:raise ValueError('label status')
  if sha(R/r['state_path'])!=r['state_sha256'] or sha(R/r['feature_path'])!=r['feature_sha256']:raise ValueError('artifact hash')
  counts[r['dataset_source']+':'+r['split']]+=1;reasons.update(r['caption_risk_reasons']);crops+=r['cropped_for_max_10s']
 if dict(counts)!=rep['source_split_counts'] or dict(reasons)!=rep['risk_reason_counts'] or crops!=rep['cropped_count']:raise ValueError('counts')
 with np.load(sp,allow_pickle=False) as f:stats={k:f[k].copy() for k in ('feature_mean','feature_std','spatial_mean','spatial_std')}
 smoke=[]
 for split in ('train','val','test'):
  dataset=RobotMotionDataset(root=R,records=packed,split=split,**stats)
  for source in ('DexWeave/GRAB','OmniRetarget/OMOMO'):
   i=next(i for i,x in enumerate(dataset.records) if x['dataset_source']==source);item=dataset[i];motion,cond=collate_robot_motion([item])
   if not torch.isfinite(motion).all() or not torch.isfinite(cond['y']['hint']).all():raise ValueError('loader')
   smoke.append({'source':source,'split':split,'sample_id':item['sample_id'],'features':len(item['motion'])})
 out={'manifest_sha256':sha(mp),'stats_sha256':sha(sp),'samples':len(packed),'counts':dict(counts),'crops':crops,
      'loader_smoke':smoke,'technical_format_passed':True,'semantics_verified':False,'source_use_scope_verified':False,
      'training_ready':False,'limitations':['Checks preserve published caption association only, not semantic truth or rights.','No GRAB/Omni physical-action filtering.']}
 with (O/'package_audit.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
 print(json.dumps(out))
if __name__=='__main__':main()

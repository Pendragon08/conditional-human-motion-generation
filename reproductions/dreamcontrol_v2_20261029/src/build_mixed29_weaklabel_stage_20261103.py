"""Build a reversible weak-label candidate stage from the verified format preview.

No trajectory is deleted or physically screened. Exclusion from this stage means
only that a caption/window needs priority review; unflagged labels are not
semantically verified, and source-use permissions are not resolved.
"""
import hashlib,json,shutil,sys
from collections import Counter
from pathlib import Path
import numpy as np,torch
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'grab_omni_mixed29_full_preview_20261103'
T=R/'grab_omni_caption_time_triage_20261103'
O=R/'grab_omni_weaklabel_stage_20261103'
sys.path[:0]=[str(R),str(R/'robot_loader_20261030/src'),'/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen']
from prepare_omni29_training_schema_20261103 import Moments
from robot_motion_dataset import RobotMotionDataset,collate_robot_motion
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from grouped_split import audit_assignments
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1)
 if O.exists():raise FileExistsError(O)
 if shutil.disk_usage(R).free<8*1024**3+356*1024**2:raise RuntimeError('disk reserve')
 full=D/'manifest.json';triage=T/'all_candidates.jsonl';original=json.loads((D/'report.json').read_text());tr=json.loads((T/'report.json').read_text())
 if sha(full)!=original['manifest_sha256'] or sha(full)!=tr['manifest_sha256'] or sha(triage)!=tr['all_candidates_sha256']:raise ValueError('evidence hash')
 rows=json.loads(full.read_text())['records'];tri=[json.loads(x) for x in triage.read_text().splitlines()];ix={x['sample_id']:x for x in tri}
 if len(rows)!=3161 or len(ix)!=3161:raise ValueError('candidate coverage')
 staged=[];held=[]
 for r in rows:
  x=ix[r['sample_id']]
  if x['state_sha256']!=r['state_sha256'] or x['caption']!=r['caption'] or x['annotation_sha256']!=r['annotation_sha256']:raise ValueError('triage binding')
  row={**r,'label_assurance':'source_paired_weak_unreviewed','triage_reasons':x['triage_reasons'],'caption_accepted':False,'training_ready':False}
  (held if x['triage_reasons'] else staged).append(row)
 if len(staged)!=2609 or len(held)!=552:raise ValueError('triage count')
 audit_assignments(staged);counts=Counter();frames=Counter();fm,sm=Moments(263),Moments(66)
 for i,r in enumerate(staged):
  if i%64==0 and shutil.disk_usage(R).free<8*1024**3+356*1024**2:raise RuntimeError('disk reserve')
  p=R/r['feature_path']
  if sha(p)!=r['feature_sha256']:raise ValueError('feature changed')
  with np.load(p,allow_pickle=False) as f:x=np.asarray(f['features'],dtype=np.float32)
  if x.ndim!=2 or x.shape[1]!=263 or not 2<=len(x)<=199 or not np.isfinite(x).all():raise ValueError('feature shape')
  tag=r['dataset_source']+':'+r['split'];counts[tag]+=1;frames[tag]+=len(x)
  if r['split']=='train':
   with torch.no_grad():j=recover_from_ric(torch.from_numpy(x),22).numpy()
   fm.add(x);sm.add(j.reshape(len(x),66))
 if set(r['sample_id'] for r in staged)&set(r['sample_id'] for r in held):raise ValueError('overlap')
 O.mkdir(exist_ok=False)
 for name,records in [('manifest.json',staged),('priority_review.json',held)]:
  with (O/name).open('x',encoding='utf8') as f:json.dump({'records':records,'training_ready':False},f,ensure_ascii=False,allow_nan=False)
 fmean,fstd=fm.result();smean,sstd=sm.result();stats={'feature_mean':fmean,'feature_std':fstd,'spatial_mean':smean,'spatial_std':sstd}
 with (O/'stats.npz').open('xb') as f:np.savez(f,**stats)
 smoke=[]
 for split in ('train','val','test'):
  ds=RobotMotionDataset(root=R,records=staged,split=split,**stats)
  for source in ('OmniRetarget/OMOMO','DexWeave/GRAB'):
   j=next(i for i,r in enumerate(ds.records) if r['dataset_source']==source)
   x=ds[j];motion,cond=collate_robot_motion([x]);assert torch.isfinite(motion).all() and torch.isfinite(cond['y']['hint']).all()
   smoke.append({'split':split,'source':source,'sample_id':x['sample_id'],'frames':len(x['motion'])})
 report={'full_manifest_sha256':sha(full),'triage_sha256':sha(triage),'staged':len(staged),'priority_review':len(held),
  'counts':dict(counts),'frames':dict(frames),'train_frames':fm.n,'loader_smoke':smoke,
  'manifest_sha256':sha(O/'manifest.json'),'priority_review_sha256':sha(O/'priority_review.json'),'stats_sha256':sha(O/'stats.npz'),
  'training_ready':False,'caption_accepted':0,'physical_action_filter':False,
  'limitations':['A rule flag indicates review priority only, not caption wrongness; unflagged captions are weak and unreviewed.',
   'No candidate robot trajectory is removed; priority captions are retained in a separate manifest.',
   'Published source, model-compatibility and underlying data-use permissions need task-specific resolution.',
   'This stage is not an accepted or complete training dataset.']}
 with (O/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
 print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()

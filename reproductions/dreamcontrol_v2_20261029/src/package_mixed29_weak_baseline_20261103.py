"""Package the complete mixed 20 Hz/29-DoF dataset as explicitly weak labels.

This compiles existing independently audited artifacts; it does not retarget,
perform physical filtering, assert caption truth, or decide permission to train.
"""
import hashlib,json,shutil
from collections import Counter
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
F=R/'grab_omni_mixed29_full_preview_20261103'
T=R/'grab_omni_caption_time_triage_20261103'
S=R/'grab_omni_weaklabel_stage_20261103'
O=R/'grab_omni_29dof_weak_baseline_v1_20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if O.exists():raise FileExistsError(O)
 if shutil.disk_usage(R).free<8*1024**3+356*1024**2:raise RuntimeError('disk reserve')
 fr=json.loads((F/'report.json').read_text());fa=json.loads((F/'independent_audit.json').read_text());ta=json.loads((T/'report.json').read_text());sr=json.loads((S/'report.json').read_text());sa=json.loads((S/'independent_audit.json').read_text())
 manifest=F/'manifest.json';triage=T/'all_candidates.jsonl'
 if sha(manifest)!=fr['manifest_sha256'] or fa['manifest_sha256']!=fr['manifest_sha256'] or sha(triage)!=ta['all_candidates_sha256'] or ta['manifest_sha256']!=fr['manifest_sha256'] or sa['manifest_sha256']!=sr['manifest_sha256'] or sa['priority_sha256']!=sr['priority_review_sha256']:raise ValueError('input audit binding')
 rows=json.loads(manifest.read_text())['records'];review=[json.loads(x) for x in triage.read_text(encoding='utf8').splitlines()];byid={r['sample_id']:r for r in review}
 if len(rows)!=3161 or len(byid)!=3161:raise ValueError('coverage')
 flags=Counter();counts=Counter();new=[]
 for row in rows:
  t=byid[row['sample_id']]
  if any(t[k]!=row[k] for k in ('caption','annotation_sha256','state_sha256')):raise ValueError('caption or state binding')
  if t['source']!=row['dataset_source'] or t['split']!=row['split'] or t['fps_nominal']!=20.:raise ValueError('source split time')
  reasons=list(t['triage_reasons']);crop=bool(t['crop_10s'])
  for reason in reasons:flags[reason]+=1
  counts[row['dataset_source']+':'+row['split']]+=1
  new.append({**row,'label_assurance':'published_human_caption_sequence_level_weak',
     'label_semantics_verified':False,'caption_accepted':False,'training_ready':False,
     'temporal_scope':'cropped_unreviewed' if crop else 'uncropped_source_sequence_nominal',
     'caption_risk_reasons':reasons,'source_variant':t['source_variant'],
     'nominal_sample_time_s':t['nominal_sample_time_s'],
     'caption_use_note':'Source pairing only; robot semantics and any omitted events are not independently verified.'})
 if len({x['sample_id'] for x in new})!=len(new) or dict(counts)!=fr['samples_by_source_split']:raise ValueError('coverage or counts')
 O.mkdir(exist_ok=False)
 with (O/'manifest.json').open('x',encoding='utf8') as f:json.dump({'records':new,'dataset_artifact_complete':True,'training_ready':False},f,ensure_ascii=False,allow_nan=False)
 with (F/'stats.npz').open('rb') as src,(O/'stats.npz').open('xb') as dst:shutil.copyfileobj(src,dst)
 if sha(O/'stats.npz')!=fr['stats_sha256']:raise ValueError('stats mismatch')
 report={'samples':len(new),'source_split_counts':dict(counts),'risk_reason_counts':dict(flags),'cropped_count':sum(x['cropped_for_max_10s'] for x in new),
  'nominal_frame_rate_hz':20,'actuated_joint_count':29,'features':'22 keypoints / 263D; 29 joint states retained separately',
  'manifest_sha256':sha(O/'manifest.json'),'stats_sha256':sha(O/'stats.npz'),'source_manifest_sha256':sha(manifest),
  'source_triage_sha256':sha(triage),'source_full_audit_sha256':sha(F/'independent_audit.json'),
  'dataset_artifact_complete':True,'caption_semantics_accepted':0,'training_ready':False,'physical_action_filter':False,
  'limitations':['All captions are weak source-sequence labels; automatic matching and no risk flag do not establish semantic correctness.',
   '513 clips are capped at 10 seconds; their complete-source captions might describe omitted events.',
   'Omni has 312 original and 1514 modified references; modified references are not all simple global transformations.',
   'DexWeave dataset card does not specify a data license; MotionHub and GRAB upstream restrictions apply and usage scope remains unresolved.',
   'DexWeave source G1 model and target 29DoF model have not been established as identical.',
   '29-DoF states are retained separately; 263D does not losslessly encode all 29 joint angles.',
   'Group labels are nominal; external data sharing, license and semantic review are not certified.']}
 with (O/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
 print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()

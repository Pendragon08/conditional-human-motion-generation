"""Reconcile three batches and the one separately audited pilot by source index."""
from collections import Counter
import hashlib,json
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 group=R/'omomo_group_split_20261101/candidates.jsonl'; candidates=[json.loads(x) for x in group.read_text(encoding='utf8').splitlines()]
 if len(candidates)!=1826 or sha(group)!='905af6eefd88e3c06cd5ff47877cade07a9a379c1f79d46a71d227e911b31085':raise ValueError('candidate source changed')
 observed={};counts=Counter();frames=Counter();crop=Counter();variants=Counter();evidence={}
 for name in ('omni29_text_motion_pilot10_20261103','omni29_text_motion_batch_a_20261103','omni29_text_motion_batch_b_20261103'):
  d=R/name;rep=json.loads((d/'report.json').read_text());audit=json.loads((d/'audit.json').read_text());p=d/'records.jsonl';rows=[json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
  if sha(p)!=rep['records_sha256'] or audit['records_sha256']!=rep['records_sha256'] or audit['technical_failures'] or len(rows)!=rep['count'] or len(audit['cases'])!=len(rows):raise ValueError('batch audit mismatch')
  audited={x['index']:x for x in audit['cases']}
  for r in rows:
   i=r['index'];c=candidates[i]
   if i in observed or r['status']!='technical_preview' or r['source_member']!=c['motion_member'] or r['source_sha256']!=c['motion_member_sha256'] or r['annotation_sha256']!=c['annotation_sha256'] or r['caption']!=c['caption'] or r['caption_accepted'] is not False or audited[i]['artifact_sha256']!=r['artifact_sha256']:raise ValueError('batch evidence conflict')
   if sha(d/r['artifact'])!=r['artifact_sha256']:raise ValueError('changed artifact')
   observed[i]=r;counts[r['auxiliary_split']]+=1;frames[r['auxiliary_split']]+=r['feature_frames'];crop[str(r['cropped_for_max_10s'])]+=1;variants[r['variant']]+=1
  evidence[name]={'report_sha256':sha(d/'report.json'),'audit_sha256':sha(d/'audit.json'),'records_sha256':sha(p)}
 d=R/'omni29_direct_text_motion_pilot_20261103';rep=json.loads((d/'report.json').read_text());aud=json.loads((d/'audit_v2.json').read_text());c=candidates[203]
 if 203 in observed or c['motion_member']!=rep['source_member'] or c['motion_member_sha256']!=rep['source_member_sha256'] or c['caption']!=rep['caption'] or sha(d/'state_features.npz')!=rep['output_sha256'] or aud['artifact_sha256']!=rep['output_sha256']:raise ValueError('pilot evidence conflict')
 observed[203]=rep;counts[c['auxiliary_split']]+=1;frames[c['auxiliary_split']]+=rep['feature_frames'];crop['False']+=1;variants[c['variant']]+=1
 evidence[d.name]={'report_sha256':sha(d/'report.json'),'audit_sha256':sha(d/'audit_v2.json'),'artifact_sha256':sha(d/'state_features.npz')}
 if set(observed)!=set(range(1826)):raise ValueError('incomplete source index coverage')
 out={'candidate_manifest_sha256':sha(group),'total_candidates':len(observed),'split_counts_nominal':dict(counts),'feature_frames_by_split_nominal':dict(frames),'cropped_counts':dict(crop),'variant_counts':dict(variants),'accepted_captions':0,'evidence':evidence,'training_ready':False,'limitations':['Coverage and artifact checks only; source name matching is not a semantic label.','Some clips are cropped and transformed variants may not preserve captions.','Nominal split group labels do not prove independent subjects.','No author-v2 implementation or trained model is claimed; no GRAB/Omni physical-action filtering.']}
 with (R/'omni29_text_motion_coverage_audit_20261103.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
 print(json.dumps({k:out[k] for k in ('total_candidates','split_counts_nominal','feature_frames_by_split_nominal','cropped_counts','accepted_captions')}))
if __name__=='__main__':main()

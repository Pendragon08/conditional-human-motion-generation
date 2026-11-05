"""Bind candidate captions to preview windows and produce a reproducible review queue.

This only prioritizes risk; a lack of flags does not certify semantic accuracy.
No GRAB or Omni trajectory is deleted or physically screened.
"""
import hashlib,json,random,shutil
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'grab_omni_mixed29_full_preview_20261103'
O=R/'grab_omni_caption_time_triage_20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if O.exists():raise FileExistsError(O)
 if shutil.disk_usage(R).free<8*1024**3+356*1024**2:raise RuntimeError('disk reserve')
 report=json.loads((D/'report.json').read_text());manifest=D/'manifest.json'
 if sha(manifest)!=report['manifest_sha256']:raise ValueError('manifest hash')
 rows=json.loads(manifest.read_text())['records'];result=[];counts=Counter();pool=defaultdict(list)
 for r in rows:
  p=R/r['state_path']
  if sha(p)!=r['state_sha256']:raise ValueError('state mismatch')
  with np.load(p,allow_pickle=False) as f:
   ts=np.asarray(f['sample_times'],dtype=np.float64) if 'sample_times' in f else None
   if ts is None:raise ValueError('missing nominal timeline')
  if len(ts)<3 or not np.allclose(ts,np.arange(len(ts))/20.,atol=1e-9,rtol=0):raise ValueError('time grid')
  source=r['dataset_source'];crop=r['cropped_for_max_10s'];risk=r.get('caption_risk_codes',[])
  kind='original' if source.startswith('Omni') and r.get('source_member','').endswith('_original.npz') else 'augmentation' if source.startswith('Omni') else 'grab'
  labels=(['ten_second_crop'] if crop else [])+list(risk)
  review='priority_review' if labels else 'unreviewed_no_rule_flag'
  record={'sample_id':r['sample_id'],'source':source,'source_variant':kind,'split':r['split'],
          'nominal_group_id':r['source_id'],'caption':r['caption'],'candidate_captions':r.get('candidate_captions',[r['caption']]),
          'caption_source':r['caption_source'],'annotation_sha256':r['annotation_sha256'],
          'state_path':r['state_path'],'state_sha256':r['state_sha256'],'fps_nominal':20.,
          'nominal_sample_time_s':[float(ts[0]),float(ts[-1])],'sampled_states':len(ts),
          'crop_10s':crop,'triage_reasons':labels,'review_state':review,'caption_accepted':False}
  result.append(record);counts[source+'|'+review]+=1;pool[source+'|'+kind+'|'+review].append(record)
 rng=random.Random(20261103);selection=[]
 for bucket,items in sorted(pool.items()):
  # Keep nominal action groups unique inside each stratum, to avoid choosing
  # near-duplicate augmented clips in place of different source actions.
  rng.shuffle(items);seen=set();chosen=[]
  for x in items:
   if x['nominal_group_id'] in seen:continue
   seen.add(x['nominal_group_id']);chosen.append(x)
   if len(chosen)==8:break
  for x in chosen:selection.append({'stratum':bucket,**x})
 O.mkdir(exist_ok=False)
 for name,data in [('all_candidates.jsonl',result),('stratified_review_queue.jsonl',selection)]:
  with (O/name).open('x',encoding='utf8') as f:
   for x in data:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
 out={'manifest_sha256':sha(manifest),'candidates':len(result),'review_queue_size':len(selection),
      'counts':dict(counts),'review_strata_available':{k:len(v) for k,v in pool.items()},
      'review_strata_selected':dict(Counter(x['stratum'] for x in selection)),'seed':20261103,
      'all_candidates_sha256':sha(O/'all_candidates.jsonl'),'review_queue_sha256':sha(O/'stratified_review_queue.jsonl'),
      'caption_accepted':0,'training_ready':False,'limitations':['Flags prioritize review, they do not prove errors and do not establish that other captions are correct.',
      '20Hz is the nominal converted grid, not independent validation of source clock or original semantic event boundaries.',
      'Original human captions may describe events or participants not represented in robot/object states.',
      'The review queue is stratified/risk-enriched and cannot be used as an unbiased overall match-rate estimate.',
      'This creates no physical-action screen and removes no trajectory.']}
 with (O/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False)
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()

"""Cross-check archived receipts; does not rerun source/model or assess training readiness."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent.parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inventory=R/'data/bmlmovi_smplx_inventory_20261102/records.jsonl'
pilot=R/'data/bmlmovi_diverse_selection_20261102.json'; pilot_a=R/'reports/bmlmovi_full_human_audit_20261102.json'
smoke=R/'data/bmlmovi_stream_smoke_queue_20261102.json'; smoke_a=R/'reports/bmlmovi_stream_smoke_audit_20261102.json'
full=R/'data/bmlmovi_stream_full_queue_20261102.json';partial=R/'reports/bmlmovi_stream_full_partial_audit_20261103.json'
resume=R/'data/bmlmovi_stream_resume_queue_20261102.json';resume_a=R/'reports/bmlmovi_stream_resume_audit_20261103.json'
sources={x['member']:x for x in map(json.loads,inventory.read_text(encoding='utf8').splitlines())}
if len(sources)!=1801:raise ValueError('inventory coverage')
for q,a in ((smoke,smoke_a),(full,partial),(resume,resume_a)):
 if read(a)['queue_sha256']!=sha(q) or read(a)['training_ready'] is not False:raise ValueError('queue audit binding')
if read(resume)['partial_audit_sha256']!=sha(partial):raise ValueError('partial audit binding')
pilots={x['id']:x for x in read(pilot)['members']};seen={}
for case in read(pilot_a)['cases']:
 p=pilots[case['id']]
 if case['status']!='technical_sample_checked' or p['member_sha256']!=case['source_sha256']:raise ValueError('pilot mismatch')
 seen[p['member']]={'source_sha256':case['source_sha256'],'frames':case['frames'],'max_difference':case['max_abs_difference_m']}
if len(seen)!=20:raise ValueError('pilot count')
for report in (smoke_a,partial,resume_a):
 for case in read(report)['cases']:
  m=case['member']
  if case['status']!='technical_sample_checked' or m in seen:raise ValueError('failed or duplicated sample')
  seen[m]={'source_sha256':case['source_sha256'],'frames':case['frames'],'max_difference':case['max_sampled_difference_m']}
if set(seen)!=set(sources):raise ValueError('source coverage mismatch')
for member,case in seen.items():
 src=sources[member]
 if case['source_sha256']!=src['member_sha256'] or case['frames']!=src['poses_shape'][0]:raise ValueError('source hash or frame count mismatch')
out={'sources':len(seen),'source_frames':sum(x['frames'] for x in seen.values()),'max_same_model_sample_difference_m':max(x['max_difference'] for x in seen.values()),'inventory_sha256':sha(inventory),'audit_sha256':{str(p):sha(p) for p in (pilot_a,smoke_a,partial,resume_a)},'training_ready':False,'limitations':['Cross-checks recorded receipts; no independent archive reread or model rerun.','Locked-head and zero hand/face approximations; actual clock, text, robot and physics unverified.']}
with (R/'reports/bmlmovi_all_human_sources_audit_20261103.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps(out))

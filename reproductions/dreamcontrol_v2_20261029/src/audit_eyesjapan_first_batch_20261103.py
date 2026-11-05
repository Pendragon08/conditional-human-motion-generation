"""Cross-check first Eyes Japan smoke, interrupted prefix, and resumed receipts."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent.parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inv=R/'data/eyesjapan_smplx_inventory_20261102/records.jsonl'
smoke=R/'data/eyesjapan_stream_smoke_queue_20261103.json';sa=R/'reports/eyesjapan_stream_smoke_audit_20261103.json'
batch=R/'data/eyesjapan_stream_batch_queue_20261103.json';pa=R/'reports/eyesjapan_stream_batch_partial_audit_20261103.json'
resume=R/'data/eyesjapan_stream_resume_queue_20261103.json';ra=R/'reports/eyesjapan_stream_resume_audit_20261103.json'
source={x['member']:x for x in map(json.loads,inv.read_text(encoding='utf8').splitlines())};seen={}
if len(source)!=749 or read(resume)['original_queue_sha256']!=sha(batch) or read(resume)['partial_audit_sha256']!=sha(pa):raise ValueError('source or continuation mismatch')
for queue,audit in ((smoke,sa),(batch,pa),(resume,ra)):
 q=read(queue);a=read(audit)
 if a['queue_sha256']!=sha(queue) or a['training_ready'] is not False:raise ValueError('audit queue binding')
 expected={r['member']:r for r in q['members']}
 for c in a['cases']:
  m=c['member']
  if m in seen or m not in expected or c['status']!='technical_sample_checked' or c['source_sha256']!=expected[m]['source_sha256'] or m not in source or c['source_sha256']!=source[m]['member_sha256'] or c['frames']!=source[m]['poses_shape'][0]:raise ValueError('duplicate or source mismatch')
  seen[m]=c
if len(seen)!=102 or set(seen)!={r['member'] for r in read(smoke)['members']}|{r['member'] for r in read(batch)['members']}:raise ValueError('coverage')
out={'sources':len(seen),'source_frames':sum(x['frames'] for x in seen.values()),'max_same_model_sample_difference_m':max(x['max_sampled_difference_m'] for x in seen.values()),'inventory_sha256':sha(inv),'audits_sha256':{p.name:sha(p) for p in (sa,pa,ra)},'training_ready':False,'limitations':['Saved receipts cross-check, not independent archive reread or model rerun.','Locked-head and zero hand/face are approximations; source clock, text, robot and physics not accepted.']}
with (R/'reports/eyesjapan_first_batch_audit_20261103.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps(out))

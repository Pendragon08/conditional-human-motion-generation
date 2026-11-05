"""Cross-check second batch against first-batch receipts and source inventory."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
inv=R/'data/eyesjapan_smplx_inventory_20261102/records.jsonl';q=R/'data/eyesjapan_stream_batch2_queue_20261103.json';a=R/'reports/eyesjapan_stream_batch2_audit_20261103.json';prior=R/'reports/eyesjapan_first_batch_audit_20261103.json'
sources={x['member']:x for x in map(json.loads,inv.read_text(encoding='utf8').splitlines())};queue=read(q);audit=read(a);old=read(prior)
excluded={x['member'] for p in ('eyesjapan_stream_smoke_queue_20261103.json','eyesjapan_stream_batch_queue_20261103.json') for x in read(R/'data'/p)['members']}
expected={x['member']:x for x in queue['members']};cases={x['member']:x for x in audit['cases']}
if len(sources)!=749 or len(excluded)!=102 or len(expected)!=97 or set(expected)&excluded or set(expected)!=set(cases) or len(cases)!=len(audit['cases']) or old['sources']!=102 or queue['prior_audit_sha256']!=sha(prior) or queue['inventory_sha256']!=sha(inv) or audit['queue_sha256']!=sha(q) or audit['training_ready'] is not False:raise ValueError('coverage or binding mismatch')
for m,c in cases.items():
 if c['status']!='technical_sample_checked' or c['source_sha256']!=expected[m]['source_sha256'] or c['source_sha256']!=sources[m]['member_sha256'] or c['frames']!=sources[m]['poses_shape'][0]:raise ValueError('source mismatch')
out={'sources':len(cases),'source_frames':sum(c['frames'] for c in cases.values()),'max_same_model_sample_difference_m':max(c['max_sampled_difference_m'] for c in cases.values()),'prior_audit_sha256':sha(prior),'batch_audit_sha256':sha(a),'training_ready':False,'limitations':['Cross-checks recorded receipts, not independent archive/model rerun.','Actual clock, text, robot, contact and physics unverified.']}
with (R/'reports/eyesjapan_second_batch_audit_20261103.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps(out))

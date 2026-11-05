"""Resume only source members not checked in the interrupted batch prefix."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent.parent
queue=R/'data/eyesjapan_stream_batch_queue_20261103.json';audit=R/'reports/eyesjapan_stream_batch_partial_audit_20261103.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
q=json.loads(queue.read_text(encoding='utf8'));a=json.loads(audit.read_text(encoding='utf8'));expected={r['member']:r for r in q['members']};cases={r['member']:r for r in a['cases']}
if q['training_ready'] is not False or a['training_ready'] is not False or a['queue_sha256']!=sha(queue) or a['expected_count']!=len(expected) or len(cases)!=len(a['cases']) or not set(cases).issubset(expected):raise ValueError('invalid audit binding')
passed={m for m,r in cases.items() if r['status']=='technical_sample_checked' and r['source_sha256']==expected[m]['source_sha256']}
remaining=[r for r in q['members'] if r['member'] not in passed]
if not remaining or len(passed)+len(remaining)!=len(expected):raise ValueError('invalid remainder')
out=R/'data/eyesjapan_stream_resume_queue_20261103.json'
with out.open('x',encoding='utf8') as f:json.dump({'members':remaining,'original_queue_sha256':sha(queue),'partial_audit_sha256':sha(audit),'training_ready':False},f,ensure_ascii=False,indent=2)
print(json.dumps({'audited_passed':len(passed),'remaining':len(remaining),'failed_audit_cases':len(cases)-len(passed)}))

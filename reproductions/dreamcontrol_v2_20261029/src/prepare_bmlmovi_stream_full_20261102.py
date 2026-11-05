import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
full=root/'data/bmlmovi_full_stream_queue_20261102.json';smoke=root/'data/bmlmovi_stream_smoke_queue_20261102.json';audit=root/'reports/bmlmovi_stream_smoke_audit_20261102.json'
q=json.loads(full.read_text(encoding='utf8'));s=json.loads(smoke.read_text(encoding='utf8'));a=json.loads(audit.read_text(encoding='utf8'))
assert a['queue_sha256']==sha(smoke) and a['training_ready'] is False
passed={x['member']:x['source_sha256'] for x in a['cases'] if x['status']=='technical_sample_checked'}
assert passed=={x['member']:x['source_sha256'] for x in s['members']} and len(passed)==2
remaining=[r for r in q['members'] if r['member'] not in passed]
assert len(remaining)==1779 and q['training_ready'] is False
out=root/'data/bmlmovi_stream_full_queue_20261102.json'
with out.open('x',encoding='utf8') as f:json.dump({'members':remaining,'original_queue_sha256':sha(full),'smoke_audit_sha256':sha(audit),'training_ready':False},f,ensure_ascii=False,indent=2)

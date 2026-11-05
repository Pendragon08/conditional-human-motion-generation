"""Build a continuation only after validating the interrupted prefix audit."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent.parent
qfile=root/'data/bmlmovi_stream_full_queue_20261102.json'
afile=root/'reports/bmlmovi_stream_full_partial_audit_20261103.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
q=json.loads(qfile.read_text(encoding='utf8'));a=json.loads(afile.read_text(encoding='utf8'))
expected={x['member']:x for x in q['members']};cases=a['cases'];seen={x['member']:x for x in cases}
if (q['training_ready'] is not False or a['training_ready'] is not False or a['queue_sha256']!=sha(qfile)
    or a['expected_count']!=len(expected) or len(cases)!=len(seen) or not set(seen).issubset(expected)):
 raise ValueError('audit or source queue binding mismatch')
passed={m for m,c in seen.items() if c['status']=='technical_sample_checked' and c['source_sha256']==expected[m]['source_sha256']}
remaining=[r for r in q['members'] if r['member'] not in passed]
if len(passed)+len(remaining)!=len(expected) or not remaining:raise ValueError('nothing or inconsistent to resume')
out=root/'data/bmlmovi_stream_resume_queue_20261102.json'
with out.open('x',encoding='utf8') as f:json.dump({'members':remaining,'original_queue_sha256':sha(qfile),'partial_audit_sha256':sha(afile),'accepted_prefix_for_reprocessing_exclusion':len(passed),'training_ready':False},f,ensure_ascii=False,indent=2)
print(json.dumps({'audited_passed':len(passed),'resume_members':len(remaining),'failed_audit_cases':len(seen)-len(passed)}))

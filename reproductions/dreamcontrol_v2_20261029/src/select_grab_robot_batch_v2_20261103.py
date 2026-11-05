"""Select 60 audited human sources from the second GRAB batch for G1 conversion."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records=R/'reports/grab_full_human_source_records_v2_20261102.jsonl';report=R/'reports/grab_full_human_source_batch_v2_20261102.json';audit=R/'reports/grab_full_human_source_audit_v2_20261102.json'
rows=[json.loads(x) for x in records.read_text(encoding='utf8').splitlines()];r=json.loads(report.read_text());a=json.loads(audit.read_text());cases={x['member']:x for x in a['cases']}
if len(rows)!=180 or len(cases)!=180 or r['records_sha256']!=sha(records) or a['batch_report_sha256']!=sha(report):raise ValueError('source evidence mismatch')
for i,row in enumerate(rows):
 c=cases[row['member']]
 if row['selection_index']!=i or row['status']!='reconstructed_unreviewed' or c['status']!='technical_sample_checked' or c['artifact_sha256']!=row['artifact_sha256']:raise ValueError('source incomplete')
chosen=sorted(rows,key=lambda x:(x['frames'],x['selection_index']))[:60]
out={'samples':[f"{x['selection_index']:03d}" for x in chosen],'source_members':[x['member'] for x in chosen],
     'source_records_sha256':sha(records),'source_report_sha256':sha(report),'source_audit_sha256':sha(audit),
     'frames':sum(x['frames'] for x in chosen),'training_ready':False,'rule':'60 shortest among previously audited second-batch clips, not random'}
with (R/'data/grab_robot_batch_v2_selection_20261103.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps({'samples':len(chosen),'frames':out['frames']}))

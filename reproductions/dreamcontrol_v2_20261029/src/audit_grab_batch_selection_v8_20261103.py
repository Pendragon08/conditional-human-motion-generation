"""Cross-check actual GRAB batch receipts and all three prior selections."""
import collections,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
reg=R/'reports/grab_source_registry_20261030.jsonl';sel=R/'data/grab_full_human_selection_v8_20261103.json'
recs=R/'reports/grab_full_human_source_records_v8_20261103.jsonl';rep=R/'reports/grab_full_human_source_batch_v8_20261103.json';audit=R/'reports/grab_full_human_source_audit_v8_20261103.json'
priors=[R/'data'/n for n in ('grab_full_human_selection_20261102.json','grab_full_human_selection_v2_20261102.json','grab_full_human_selection_v3_20261102.json','grab_full_human_selection_v4_20261102.json','grab_full_human_selection_v5_20261103.json','grab_full_human_selection_v6_20261103.json','grab_full_human_selection_v7_20261103.json')]
s=json.loads(sel.read_text(encoding='utf8'));p=json.loads(rep.read_text());a=json.loads(audit.read_text());rows=[json.loads(x) for x in recs.read_text().splitlines()];sources={x['member']:x for x in map(json.loads,reg.read_text(encoding='utf8').splitlines())}
old={m for f in priors for m in json.loads(f.read_text(encoding='utf8'))['members']}
if s['source_registry_sha256']!=sha(reg) or s['prior_selection_sha256']!=[sha(f) for f in priors] or p['selection_sha256']!=sha(sel) or p['registry_sha256']!=sha(reg) or p['records_sha256']!=sha(recs) or a['batch_report_sha256']!=sha(rep):raise ValueError('bindings changed')
if [x['member'] for x in rows]!=s['members'] or len(rows)!=100 or len(set(s['members']))!=100 or set(s['members'])&old:raise ValueError('selection coverage')
cases={x['member']:x for x in a['cases']}
if len(cases)!=100 or set(cases)!=set(s['members']):raise ValueError('audit coverage')
for i,row in enumerate(rows):
 src=sources[row['member']];c=cases[row['member']]
 if row['selection_index']!=i or row['status']!='reconstructed_unreviewed' or row['source_member_sha256']!=src['member_sha256'] or row['frames']!=src['frames'] or row['declared_fps']!=src['fps_declared'] or c['status']!='technical_sample_checked' or c['source_sha256']!=src['member_sha256'] or c['artifact_sha256']!=row['artifact_sha256'] or c['full_object_array_equal'] is not True:raise ValueError('case mismatch '+row['member'])
out={'selected':100,'source_frames':sum(x['frames'] for x in rows),'subject_counts':dict(collections.Counter(sources[x['member']]['subject'] for x in rows)),'maximum_sampled_difference_m':max(x['max_sampled_joint_difference_m'] for x in cases.values()),'records_sha256':sha(recs),'audit_sha256':sha(audit),'training_ready':False,'limitations':['Cross-checks saved receipts; does not re-read source archives or prove clock, text, robot motion, contact or physics.']}
with (R/'reports/grab_full_human_selection_audit_v8_20261103.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps(out))

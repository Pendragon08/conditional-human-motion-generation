"""Select a size-bounded continuation after validating the first batch receipts."""
import hashlib,json
from collections import Counter
from pathlib import Path
R=Path(__file__).resolve().parent.parent
inventory=R/'data/eyesjapan_smplx_inventory_20261102/records.jsonl';smoke=R/'data/eyesjapan_stream_smoke_queue_20261103.json';batch=R/'data/eyesjapan_stream_batch_queue_20261103.json';audit=R/'reports/eyesjapan_first_batch_audit_20261103.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads(audit.read_text(encoding='utf8'));prev=[json.loads(p.read_text(encoding='utf8')) for p in (smoke,batch)]
excluded={r['member'] for p in prev for r in p['members']}
if a['sources']!=102 or a['inventory_sha256']!=sha(inventory) or a['training_ready'] is not False or len(excluded)!=102:raise ValueError('prior evidence mismatch')
rows=[json.loads(x) for x in inventory.read_text(encoding='utf8').splitlines()];selected=[];frames=0;subjects=Counter()
if len(rows)!=749 or len({r['member'] for r in rows})!=749:raise ValueError('inventory')
for r in sorted(rows,key=lambda x:(x['poses_shape'][0],x['member'])):
 subject=r['member'].split('/')[1]
 if r['member'] in excluded or subjects[subject]>=10:continue
 if r.get('training_accepted') is not False or r.get('duplicate_key') or r['declared_fps']!=120. or r['member_size']>64*1024**2:raise ValueError('invalid source')
 if (frames+r['poses_shape'][0])*300>100*1024**2:continue
 selected.append({'member':r['member'],'source_sha256':r['member_sha256'],'member_size':r['member_size'],'source_frames':r['poses_shape'][0],'declared_fps':r['declared_fps'],'nominal_rows':r['nominal_rows']});frames+=r['poses_shape'][0];subjects[subject]+=1
 if len(selected)==100:break
if not selected:raise ValueError('empty selection')
out=R/'data/eyesjapan_stream_batch2_queue_20261103.json'
with out.open('x',encoding='utf8') as f:json.dump({'members':selected,'inventory_sha256':sha(inventory),'prior_audit_sha256':sha(audit),'raw_estimate_300_bytes_per_frame':frames*300,'training_ready':False,'rule':'shorter sources, at most 10 per nominal subject, nonrandom'},f,ensure_ascii=False,indent=2)
print(json.dumps({'members':len(selected),'frames':frames,'estimated_bytes':frames*300,'subjects':len(subjects)}))

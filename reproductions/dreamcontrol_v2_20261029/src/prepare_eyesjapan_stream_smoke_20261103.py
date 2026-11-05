"""Select two named-source examples for bounded streaming reconstruction."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent.parent
p=R/'data/eyesjapan_smplx_inventory_20261102/records.jsonl'; summary=R/'data/eyesjapan_smplx_inventory_20261102/summary.json'
def sha(x):return hashlib.sha256(x.read_bytes()).hexdigest()
if json.loads(summary.read_text())['event']!='complete':raise ValueError('inventory incomplete')
rows=[json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]
if len(rows)!=749 or len({x['member'] for x in rows})!=749:raise ValueError('inventory coverage')
by_subject={}
for r in rows:
 if r.get('duplicate_key') or r.get('training_accepted') is not False or r.get('declared_fps')!=120. or r['member_size']>64*1024**2:raise ValueError('invalid candidate')
 subject=r['member'].split('/')[1]
 if subject not in by_subject or r['poses_shape'][0]<by_subject[subject]['poses_shape'][0]:by_subject[subject]=r
selected=sorted(by_subject.values(),key=lambda x:(x['poses_shape'][0],x['member']))[:2]
if len(selected)!=2:raise ValueError('insufficient subjects')
members=[{'member':r['member'],'source_sha256':r['member_sha256'],'member_size':r['member_size'],'source_frames':r['poses_shape'][0],'declared_fps':r['declared_fps'],'nominal_rows':r['nominal_rows']} for r in selected]
out=R/'data/eyesjapan_stream_smoke_queue_20261103.json'
with out.open('x',encoding='utf8') as f:json.dump({'members':members,'inventory_sha256':sha(p),'training_ready':False,'rule':'two shortest candidates from different nominal subject folders, not a random selection'},f,ensure_ascii=False,indent=2)
print(json.dumps({'members':len(members),'frames':sum(r['source_frames'] for r in members)}))

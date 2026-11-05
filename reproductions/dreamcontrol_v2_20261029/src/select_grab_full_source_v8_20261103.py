"""Select a small, bounded continuation from the existing GRAB registry."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
registry=ROOT/'reports/grab_source_registry_20261030.jsonl'
priors=[ROOT/'data'/n for n in ('grab_full_human_selection_20261102.json','grab_full_human_selection_v2_20261102.json','grab_full_human_selection_v3_20261102.json','grab_full_human_selection_v4_20261102.json','grab_full_human_selection_v5_20261103.json','grab_full_human_selection_v6_20261103.json','grab_full_human_selection_v7_20261103.json')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prior_data=[json.loads(p.read_text(encoding='utf8')) for p in priors]
if any(x['training_ready'] is not False or x['source_registry_sha256']!=sha(registry) for x in prior_data):raise ValueError('prior evidence mismatch')
excluded={m for s in prior_data for m in s['members']}
excluded.update({'s3/apple_eat_1.npz','s5/cup_pour_1.npz','s4/waterbottle_drink_1.npz'})
rows=[json.loads(line) for line in registry.read_text(encoding='utf8').splitlines()]
chosen=[]
for subject in [f's{i}' for i in range(1,11)]:
 candidates=[r for r in rows if r['subject']==subject and r['member'] not in excluded and 240<=r['frames']<=1800 and r['member_bytes']<=8_000_000 and r['fps_declared']==120.0]
 candidates.sort(key=lambda r:(r['frames'],r['member']))
 chosen.extend(candidates[:10])
if len(chosen)<100:
 used=excluded|{r['member'] for r in chosen}
 extra=[r for r in rows if r['member'] not in used and 240<=r['frames']<=1800 and r['member_bytes']<=8_000_000 and r['fps_declared']==120.0]
 extra.sort(key=lambda r:(r['frames'],r['member']))
 chosen.extend(extra[:100-len(chosen)])
members=[r['member'] for r in chosen]
if len(members)!=100 or len(set(members))!=100 or set(members)&excluded:raise ValueError('selection duplication')
frames=sum(r['frames'] for r in chosen)
selection={'members':members,'source_registry_sha256':sha(registry),'prior_selection_sha256':[sha(p) for p in priors],
 'selected_source_frames':frames,'estimated_conservative_600B_per_frame':600*frames,'training_ready':False,
 'rule':'up to 10 shortest eligible per subject, then shortest remaining to reach 100, excluding prior selections and three earlier individual examples; deterministic, not random'}
out=ROOT/'data/grab_full_human_selection_v8_20261103.json'
with out.open('x',encoding='utf8') as f:json.dump(selection,f,ensure_ascii=False,indent=2)
print(json.dumps({'members':len(members),'frames':frames,'estimated_bytes':600*frames}))

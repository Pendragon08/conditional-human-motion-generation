import json,time
from pathlib import Path
p=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/results/physics_guided_20261011')
j=p/'queue_pause_journal.json'; original=j.read_text(); (p/'queue_pause_journal_original.json').write_text(original)
d=json.loads(original); d['original_reason_inaccurate']=d['reason']; d['reason']='Second pause during test_eval_random_best_s11 for pre-ablation integrity fixes; proposed_s11_r1/r2 had already completed with actual prefixes. Original wording incorrectly described timing.';d['corrected_at']=time.time();j.write_text(json.dumps(d,indent=2))
h=[{'event':'first_pause','child_label':'test_planned_s11_r2','queue_pid':2589112,'reason':'Align proposed mechanism with actual execution prefix before any proposed candidate run','evidence':'budget ledger + metadata for proposed candidates','timestamp_not_recovered':True},{'event':'second_pause','state':d['state'],'queue_pid':d['pid'],'time':d['paused'],'reason':d['reason'],'original_journal_preserved':'queue_pause_journal_original.json'}]
(p/'queue_pause_history.json').write_text(json.dumps(h,indent=2));(p/'queue_state.json').write_text(json.dumps({'status':'PAUSED_WAITING_DOCUMENT_WITNESS_V4','time':time.time()},indent=2))
print('PROVENANCE_CORRECTION_PRESERVES_ORIGINAL_JOURNAL')


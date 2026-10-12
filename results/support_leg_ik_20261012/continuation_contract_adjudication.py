"""Separate read-only contract adjudication; preserves the first continuation FAIL."""
import hashlib
import json
import time
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
OUT=ROOT/'results/support_leg_ik_20261012'
PARENT=ROOT/'results/support_reference_probe_v63_20261012'
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
w=read(OUT/'CONTINUATION_DOCUMENT_WITNESS.json');old=read(OUT/'DOCUMENT_WITNESS.json')
add=read(PARENT/'NUMERICAL_CONTRACT_ADDENDUM.json');ledger=read(PARENT/'budget_ledger.json');summary=read(PARENT/'summary.json')
r=dict(status='FAIL',independent_witness=True,witness_agent='/root/support_leg_ik_continuation_follower',
 time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),preflight_sha256=w['preflight_sha256'],
 source_sha256=w['source_sha256'],document_sha256=w['document_sha256'],actual_terminal=w['actual_terminal'],
 original_failure_preserved=True,continuation_FAIL_preserved=True,checks={},failures=[],inspected_hashes={},
 check_invocations_this_adjudication=0,run_invocations=0,simulation_calls=0,generation_calls=0,training_calls=0,
 api_calls=0,retries=0,source_changes=0,thresholds_changed=False,simulation_authorization=False,
 explanation='Initial continuation audit incorrectly required the historical whole parent budget ledger to remain byte-identical after completed append-only parent arm runs. The documented contract requires cached baseline bytes and its original cost retained; all other first audit checks passed.',
 limitations=w['limitations']+['This separate adjudication does not rewrite or satisfy the runner gate requiring CONTINUATION_DOCUMENT_WITNESS.status=PASS.'])
def check(name,value,detail=None):
    r['checks'][name]=bool(value)
    if not value:r['failures'].append(dict(check=name,detail=detail))
def inspect(p):
    h=sha(p);r['inspected_hashes'][str(p.relative_to(ROOT))]=h;return h
check('first_continuation_FAIL_preserved',w['status']=='FAIL' and w['failures']==[dict(check='cached_parent_budget_ledger.json',detail=None)])
check('all_documented_first_audit_checks_pass',all(v for k,v in w['checks'].items() if k!='cached_parent_budget_ledger.json'))
for n,h in w['inspected_hashes'].items():check('first_inspected_input_unchanged_'+n,inspect(ROOT/n)==h)
for n,h in add['input_hashes'].items():
    if n!='budget_ledger.json':check('historical_parent_input_'+n,inspect(PARENT/n)==h)
r['bindings']=dict(continuation_FAIL_sha256=inspect(OUT/'CONTINUATION_DOCUMENT_WITNESS.json'),
 continuation_FAIL_md_sha256=inspect(OUT/'CONTINUATION_DOCUMENT_WITNESS.md'),
 original_FAIL_sha256=inspect(OUT/'DOCUMENT_WITNESS.json'),original_FAIL_md_sha256=inspect(OUT/'DOCUMENT_WITNESS.md'),
 parent_addendum_sha256=inspect(PARENT/'NUMERICAL_CONTRACT_ADDENDUM.json'),
 parent_summary_sha256=inspect(PARENT/'summary.json'),parent_ledger_sha256=inspect(PARENT/'budget_ledger.json'))
check('original_FAIL_unchanged',old['status']=='FAIL' and r['bindings']['original_FAIL_sha256']==read(OUT/'SCALAR_CONTRACT_DIAGNOSIS.json')['bindings']['original_files']['DOCUMENT_WITNESS.json'])
baseline=[x for x in ledger if x['label']=='val_baseline_s11'];original_charge=read(PARENT/'DOCUMENT_WITNESS.json')['budget']['charged_baseline_ledger']
historical_serialized=hashlib.sha256(json.dumps(baseline,indent=2).encode()).hexdigest()
check('cached_baseline_original_charge_exact',baseline==original_charge==w['cached_baseline_original_charge'])
check('historical_ledger_was_baseline_only',historical_serialized==add['input_hashes']['budget_ledger.json'])
check('ledger_append_only_completed_arms',[x['label'] for x in ledger]==['val_baseline_s11','val_unconstrained_s11','val_endpoint_preserved_s11'] and summary['status']=='COMPLETE_REFERENCE_SIDE_DIAGNOSIS')
for row in ledger:
    arm=row['label'][4:-4];directory=PARENT/'raw'/row['label'];wanted=PARENT/'requests'/(row['label']+'.json')
    lin=read(directory/'lineage.json');rec=read(directory/'receipt.json');report=summary['reports'][arm]
    check(arm+'_charge_matches_completed_request',row['exit_code']==0 and row['seconds']>0 and row['request_sha256']==inspect(wanted) and row['stage']==('measurement' if arm=='baseline' else 'comparison'))
    check(arm+'_bound_complete_receipt',lin['request']==read(wanted) and lin['certification']=='bound_request_at_execution' and rec['status']=='complete' and set(lin['outputs'])=={'metadata.json','metrics.json','trajectories.npz'} and report['cases']==72)
    for n,h in lin['outputs'].items():check(arm+'_'+n,inspect(directory/n)==h==report['binding']['output_hashes'][n])
    check(arm+'_summary_lineage',inspect(directory/'lineage.json')==report['binding']['lineage_sha256'] and inspect(directory/'receipt.json')==report['binding']['receipt_sha256'] and rec['trajectories_sha256']==lin['outputs']['trajectories.npz'])
r['ledger_observation']=dict(historical_whole_ledger_sha256=add['input_hashes']['budget_ledger.json'],current_whole_ledger_sha256=r['bindings']['parent_ledger_sha256'],historical_baseline_only_serialization_sha256=historical_serialized,rows=ledger,charged_seconds=sum(x['seconds'] for x in ledger),meaning='Expected append-only accounting for two subsequently completed parent arms; original baseline row unchanged.')
r['reference_invariants']=w['reference_invariants'];r['original_witness_sha256']=r['bindings']['continuation_FAIL_sha256'];r['input_hashes']=dict(r['inspected_hashes']);r['status']='FAIL' if r['failures'] else 'PASS_DOCUMENTED_CONTINUATION_INVARIANTS'
with (OUT/'CONTINUATION_CONTRACT_ADDENDUM.json').open('x') as f:json.dump(r,f,indent=2)
lines=['# Separate continuation contract adjudication','','Status: '+r['status'],'','First CONTINUATION_DOCUMENT_WITNESS FAIL and original DOCUMENT_WITNESS FAIL remain untouched. No --check retry, simulator, generation, training, source modification or threshold change.','','The first continuation audit added an undocumented whole-parent-ledger equality criterion. Its historical hash is reproduced exactly from the unchanged baseline-only row. The current ledger appends the two completed parent arms; each charge is bound to its original request, complete receipt, lineage and raw outputs. All documented preparation checks in the first audit passed.','']
lines += [k+': '+r[k] for k in ('preflight_sha256','source_sha256','document_sha256')]
lines += ['','This adjudication supplies no simulator authorization and does not satisfy the current runner gate requiring the untouched continuation witness itself to be PASS.','','Limitations:']+['- '+x for x in r['limitations']]+['','Failures: '+json.dumps(r['failures'])]
with (OUT/'CONTINUATION_CONTRACT_ADDENDUM.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(dict(status=r['status'],bindings=r['bindings'],ledger_observation=r['ledger_observation'],failures=r['failures']),indent=2))

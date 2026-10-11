"""Numeric launcher integration fixtures only; no physics, generation or efficacy."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import run_condition_feedback_guard_v31_20261012 as run

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/condition_feedback_guard_launcher_numeric_v31_20261012'
OUT.mkdir(exist_ok=True)
run.OUT=OUT
config=json.loads((ROOT/'configs/physics_guided_20261011.json').read_text(encoding='utf8'))
checks={}
previous_read=run.read
run.read=lambda path: dict(status='RUNNING') if Path(path)==run.BASE/'queue_state.json' else previous_read(path)
try:
    run.prepare()
    raise AssertionError('Live C1 must not allow C2 launch')
except AssertionError as error:
    checks['live_c1_blocks_launcher']='Wait for the live C1' in str(error)
finally:
    run.read=previous_read

run.read=lambda path: dict(entries=[],directories=[]) if Path(path)==OUT/'prior_budget_snapshot.json' else previous_read(path)
try:
    namespace=run.queue_namespace()
    checks['all_inner_size_polls_use_cumulative_output']='cumulative_output_bytes' in namespace['Queue'].run.__code__.co_names
    checks['cached_and_new_paths_use_strict_guard_wrapper']=namespace['Queue'].run.__globals__['certify_cached'] is namespace['certify_cached']
    checks['frozen_c2_collector_shell_selected']=namespace['SHELL'].name=='physics_guided_run_v31_20261012.sh'
finally:
    run.read=previous_read

cases=[];first=[];repeat=[]
for family in config['families']:
    for source in range(2):
        for offset in (-.15,0,.15):
            case=dict(case_id=family+str(source)+':'+str(offset),source_id=family+str(source),family=family)
            cases.append(case)
            row=dict(case,completed=True,length=120,endpoint_error_vector=[.1,0,0],
                heading_error_signed_rad=.1,endpoint_error_m=.1,heading_error_rad=.1,
                endpoint_time_error_s=0.,surface_slip_proxy_fraction=.02,
                penetration_fraction=0.,jerk_p95_m_s3=10.,tracking_mean_m=.01,tracking_p95_m=.02)
            first.append(row);second=copy.deepcopy(row)
            if source==1: second['endpoint_error_vector']=[.2,0,0]
            repeat.append(second)
n=len(cases)
meta=dict(cases=cases,original_goal=np.zeros((n,3)).tolist(),proposed_goal=np.zeros((n,3)).tolist(),
    original_heading=np.zeros(n).tolist(),proposed_heading=np.zeros(n).tolist(),
    context_steps=60,original_goal_sha256='synthetic-fixture-not-a-real-target-hash',
    generation_noise_sha256='synthetic-fixture-not-a-real-noise-hash')
class Arrays(dict):
    def close(self): pass
first_path=OUT/'fixture_first';repeat_path=OUT/'fixture_repeat'
for path,rows in ((first_path,first),(repeat_path,repeat)):
    path.mkdir(exist_ok=True)
    for filename,value in (('metadata.json',meta),('metrics.json',rows)):
        (path/filename).write_text(json.dumps(value),encoding='utf8')
    # An explicitly synthetic sentinel sufficient for binding tests, not a NPZ.
    (path/'trajectories.npz').write_bytes(b'synthetic fixture sentinel; no trajectory')
def load(path):
    return copy.deepcopy(meta),copy.deepcopy(first if path==first_path else repeat),Arrays()
namespace['load']=load
namespace['CONFIG']=config
for variant in ('proposed','no_agreement','no_quality_guard','shuffled_agreement'):
    output=run.feedback(namespace,first_path,repeat_path,variant,'fixture_'+variant)
    value=previous_read(output)
    checks[variant+'_keeps_original_target']=value['original_goal_sha256']==meta['original_goal_sha256']
    checks[variant+'_bounded_position']=bool(np.max(np.linalg.norm(value['proposed_goal'],axis=1))<=.050001)
    if variant=='shuffled_agreement':
        checks['shuffled_labels_cross_source_same_family_same_offset']=all(
            cases[i]['source_id']!=cases[j]['source_id'] and cases[i]['family']==cases[j]['family']
            and cases[i]['case_id'].split(':')[1]==cases[j]['case_id'].split(':')[1]
            for i,j in enumerate(value['agreement_donor_indices']))
    if variant=='proposed':
        original=value
    if variant=='no_agreement':
        checks['no_agreement_changes_disagreeing_components']=value['proposed_goal']!=original['proposed_goal']
checks['repeat_outputs_explicitly_bound']=all(name in value['repeat_outputs'] for name in ('metadata.json','metrics.json','trajectories.npz'))
repeat[0]['completed']=False
invalid=previous_read(run.feedback(namespace,first_path,repeat_path,'shuffled_agreement','fixture_invalid_local'))
checks['shuffled_labels_do_not_license_invalid_local_feedback']=bool(
    np.array_equal(invalid['proposed_goal'][0],meta['original_goal'][0]) and
    invalid['proposed_heading'][0]==meta['original_heading'][0])
sources=('scripts/run_condition_feedback_guard_v31_20261012.py',
    'scripts/analyze_condition_feedback_guard_v31_20261012.py',
    'scripts/check_condition_feedback_guard_launcher_v31_20261012.py',
    'scripts/condition_feedback_guard_core_v29_20261012.py',
    'docs/condition_feedback_guard_simulation_v31_20261012.md')
result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,
    evaluation_type='synthetic numeric launcher integration fixtures',simulation_calls=0,
    generation_calls=0,simulation_efficacy_tested=False,innovation_confirmed=False,
    source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
(OUT/'integration_check.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2));assert all(checks.values()),checks

"""Descriptive frozen-criterion decomposition; never creates alternative scores."""
import json
import math
from pathlib import Path
from child_input_guard_20261012 import sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/controller_state_diagnosis_v2_20261012'

def main():
    t=json.loads((OUT/'frozen_thresholds.json').read_text())['thresholds']
    bindings={'thresholds':sha(OUT/'frozen_thresholds.json')};reports={}
    pairs={'position':('endpoint_error_m','goal_position_m',0),
        'heading':('heading_error_rad','heading_rad',0),
        'time_proxy':('endpoint_time_error_s','goal_time_s',1e-6),
        'penetration_proxy':('penetration_fraction','penetration_fraction',0),
        'surface_slip_proxy':('surface_slip_proxy_fraction','sliding_fraction',0),
        'continuity_proxy':('jerk_p95_m_s3','continuity_jerk_p95_m_s3',0),
        'tracking_mean_proxy':('tracking_mean_m','tracking_mean_m',0),
        'tracking_p95_proxy':('tracking_p95_m','tracking_p95_m',0)}
    for state in ('multitask','t2m','unfinetuned'):
        path=OUT/'raw'/state/'metrics.json';rows=json.loads(path.read_text());assert len(rows)==72
        bindings[state]=sha(path);failures={k:0 for k in pairs};physical_rows=[]
        for r in rows:
            failed=[]
            for name,(metric,threshold,tol) in pairs.items():
                value=r.get(metric)
                if value is None or not math.isfinite(value) or value>t[threshold]+tol:
                    failures[name]+=1;failed.append(name)
            if not r['completed']:failed.append('completion')
            if r['physical_pass']:
                physical_rows.append(dict(case_id=r['case_id'],source_id=r['source_id'],
                    failed_frozen_criteria=failed,endpoint_error_m=r['endpoint_error_m'],
                    heading_error_rad=r['heading_error_rad'],endpoint_time_error_s=r['endpoint_time_error_s'],
                    joint_pass=r['joint_pass']))
        reports[state]=dict(inputs=72,criterion_failure_counts=failures,
            completion_failures=sum(not r['completed'] for r in rows),physically_passing_rows=physical_rows)
    result=dict(status='DESCRIPTIVE_CRITERION_BREAKDOWN',reports=reports,input_hashes=bindings,
        source_sha256=sha(Path(__file__)),new_simulator_batches=0,generation_calls=0,
        thresholds_changed=False,alternative_success_rule=False,candidate_efficacy_test=False,
        scope='Overlapping failure counts, frozen endpoint-near-minimum timing and physics proxies; no relaxed gate or causal claim')
    with (OUT/'failure_components.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(reports,indent=2),flush=True)
    print('CONTROLLER_STATE_CRITERION_BREAKDOWN_COMPLETE',flush=True)

if __name__=='__main__':main()

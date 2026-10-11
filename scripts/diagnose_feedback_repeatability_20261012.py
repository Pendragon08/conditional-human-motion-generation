"""Read-only C2 mechanism diagnostic from existing C1 zero-update repeats.

This is not the C2 three-candidate simulation or its quality-guard ablation.
No generated motion, new simulation or efficacy verdict is produced.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from condition_feedback_guard_core_v29_20261012 import agreement, corrected_conditions

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/condition_feedback_formal_v28_20261012'
OUT=ROOT/'results/feedback_repeatability_diagnostic_20261012'

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def load(label):
    p=BASE/'raw'/label
    lineage=read(p/'lineage.json');receipt=read(p/'receipt.json')
    assert receipt['status']=='complete' and receipt['cases']==144
    assert all(sha(p/name)==digest for name,digest in lineage['outputs'].items())
    return p,read(p/'metadata.json'),read(p/'metrics.json'),lineage

def main():
    config=read(ROOT/'configs/physics_guided_20261011.json');thresholds=read(BASE/'frozen_thresholds.json')['thresholds']
    assert config['base_thresholds']==thresholds
    results=[]
    for seed in (11,12,13):
        first_label='test_native_s%d_r0'%seed
        if not (BASE/'raw'/first_label/'lineage.json').exists():continue
        for repeat_round in (1,2):
            repeat_label='test_no_feedback_s%d_r%d'%(seed,repeat_round)
            if not (BASE/'raw'/repeat_label/'lineage.json').exists():continue
            a,am,ar,al=load(first_label);b,bm,br,bl=load(repeat_label)
            assert am['cases']==bm['cases']
            for field in ('original_goal','original_heading'):
                assert am[field]==bm[field],field
            input_differences={field:dict(exactly_equal=am[field]==bm[field],
                maximum_absolute_delta=float(np.max(np.abs(np.asarray(am[field])-np.asarray(bm[field])))))
                for field in ('proposed_goal','proposed_heading')}
            assert al['request']['noise_round']==bl['request']['noise_round']==0
            fields={}
            with np.load(a/'trajectories.npz',allow_pickle=False) as x,np.load(b/'trajectories.npz',allow_pickle=False) as y:
                assert x.files==y.files
                for key in x.files:
                    assert x[key].shape==y[key].shape
                    fields[key]=bool(np.array_equal(x[key],y[key]))
            reliability=[agreement(x,y,thresholds) for x,y in zip(ar,br)]
            guarded=corrected_conditions(am,ar,br,config,use_agreement=True)
            plain=corrected_conditions(am,ar,br,config,use_agreement=False)
            goal_delta=np.linalg.norm(np.asarray(guarded['proposed_goal'])-np.asarray(plain['proposed_goal']),axis=1)
            heading_delta=np.abs(np.asarray(guarded['proposed_heading'])-np.asarray(plain['proposed_heading']))
            completed=np.array([r['repeat_completed'] for r in reliability])
            results.append(dict(seed=seed,first=first_label,repeat=repeat_label,attempts=144,
                proposed_condition_record_differences=input_differences,
                exact_proposed_conditions_match=all(row['exactly_equal'] for row in input_differences.values()),
                repeated_complete=int(completed.sum()),position_agreement=sum(r['position'] for r in reliability),
                heading_agreement=sum(r['heading'] for r in reliability),exact_array_equality=fields,
                all_arrays_equal=all(fields.values()),
                guarded_vs_plain_goal_changed_cases=int((goal_delta>0).sum()),
                guarded_vs_plain_heading_changed_cases=int((heading_delta>0).sum()),
                completed_cases_goal_changes=int(((goal_delta>0)&completed).sum()),
                completed_cases_heading_changes=int(((heading_delta>0)&completed).sum()),
                maximum_goal_delta_m=float(goal_delta.max()),maximum_heading_delta_rad=float(heading_delta.max()),
                first_bindings=al['outputs'],repeat_bindings=bl['outputs']))
    assert results,'No verified completed repeated simulations available'
    OUT.mkdir(exist_ok=True)
    path=OUT/('diagnostic_'+str(time.time_ns())+'.json')
    report=dict(time=time.time(),status='COMPLETE_READ_ONLY_REPEAT_DIAGNOSTIC',entries=results,
        source_sha256=sha(Path(__file__)),core_sha256=sha(ROOT/'scripts/condition_feedback_guard_core_v29_20261012.py'),
        thresholds_sha256=sha(BASE/'frozen_thresholds.json'),numpy_version=np.__version__,
        simulation_calls=0,generation_calls=0,weights_updated=False,live_pipeline_mutated=False,
        c2_simulation_efficacy_tested=False,quality_guard_ablation_tested=False,innovation_confirmed=False,
        scope='Already-paid C1 same-noise zero-update executions; numerical proposed-condition differences retained. Not a fresh exact-input C2 repeat experiment; agreement is not accuracy or physical feasibility')
    path.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='entries'},indent=2))
    for row in results:print(row['seed'],row['repeat'],row['all_arrays_equal'],row['repeated_complete'],row['completed_cases_goal_changes'],row['completed_cases_heading_changes'])
    print('REPEAT_DIAGNOSTIC_PATH',path)

if __name__=='__main__':main()

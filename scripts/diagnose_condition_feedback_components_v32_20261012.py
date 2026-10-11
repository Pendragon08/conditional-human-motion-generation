"""Read-only failure attribution with unchanged predicates; no new success criterion.

Partial diagnostic evidence cannot establish efficacy, physical ground truth,
causality or a controller-independent impossibility. It does not change V28.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/condition_feedback_formal_v28_20261012'
OUT=ROOT/'results/condition_feedback_components_v32_20261012'
ARMS=('raw','ordinary','random_best','native_single','native_best','proposed',
      'no_feedback','shuffled','planned_prefix','tracking_only')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric(row,key,limit):
    value=row.get(key)
    return isinstance(value,(int,float)) and math.isfinite(value) and value<=limit


def flags(row,choice,thresholds):
    t=thresholds
    result=dict(position=numeric(row,'endpoint_error_m',t['goal_position_m']),
        heading=numeric(row,'heading_error_rad',t['heading_rad']),
        timing=numeric(row,'endpoint_time_error_s',t['goal_time_s']+1e-6),
        surface_slip_proxy=numeric(row,'surface_slip_proxy_fraction',t['sliding_fraction']),
        penetration=numeric(row,'penetration_fraction',t['penetration_fraction']),
        continuity=numeric(row,'jerk_p95_m_s3',t['continuity_jerk_p95_m_s3']),
        tracking_mean=numeric(row,'tracking_mean_m',t['tracking_mean_m']),
        tracking_p95=numeric(row,'tracking_p95_m',t['tracking_p95_m']),
        construction_complete=bool(choice['construction_complete']),
        replay_complete=bool(row['completed']))
    conditions=all(result[k] for k in ('position','heading','timing'))
    physical=all(result[k] for k in ('surface_slip_proxy','penetration','continuity'))
    tracking=all(result[k] for k in ('tracking_mean','tracking_p95'))
    assert conditions==row['condition_pass'] and physical==row['physical_pass'] and tracking==row['tracking_pass']
    result.update(condition=conditions,physical=physical,tracking=tracking,
        joint=bool(row['joint_pass'] and choice['construction_complete']))
    assert result['joint']==all(result[k] for k in ('condition','physical','tracking','construction_complete','replay_complete'))
    return result


def main(self_check=False):
    thresholds=read(BASE/'frozen_thresholds.json')['thresholds']
    if self_check:
        row=dict(endpoint_error_m=.1,heading_error_rad=thresholds['heading_rad'],
            endpoint_time_error_s=.1,surface_slip_proxy_fraction=.1,
            penetration_fraction=.02,jerk_p95_m_s3=1000.,tracking_mean_m=.1,
            tracking_p95_m=.2,condition_pass=True,physical_pass=True,tracking_pass=True,
            completed=True,joint_pass=True)
        assert flags(row,dict(construction_complete=True),thresholds)['joint']
        assert not flags(row,dict(construction_complete=False),thresholds)['joint']
        missing=dict(row,surface_slip_proxy_fraction=None,physical_pass=False,joint_pass=False)
        assert not flags(missing,dict(construction_complete=True),thresholds)['physical']
        print('COMPONENT_DIAGNOSTIC_NUMERIC_CHECK_PASS');return
    bindings={}; entries=[]; eligibility={}; masks={}
    for seed in (11,12,13):
        for arm in ARMS:
            path=BASE/'raw'/('test_eval_'+arm+'_s'+str(seed))
            if not (path/'receipt.json').exists() or not (path/'lineage.json').exists(): continue
            receipt=read(path/'receipt.json'); lineage=read(path/'lineage.json')
            assert receipt['status']=='complete' and receipt['cases']==144
            assert all(sha(path/name)==digest for name,digest in lineage['outputs'].items())
            rows=read(path/'metrics.json'); selected=BASE/'selected'/(arm+'_s'+str(seed)+'.json')
            choice_binding=read(selected.with_suffix('.lineage.json'))
            assert choice_binding['selection_json']==sha(selected)
            choices=read(selected)['choices']; assert len(rows)==len(choices)==144
            assert [r['case_id'] for r in rows]==[c['case_id'] for c in choices]
            components=[flags(r,c,thresholds) for r,c in zip(rows,choices)]
            passes={key:sum(f[key] for f in components) for key in components[0]}
            failures=Counter(','.join(k for k in ('position','heading','timing',
                'surface_slip_proxy','penetration','continuity','tracking_mean',
                'tracking_p95','construction_complete','replay_complete') if not f[k])
                or 'all_pass' for f in components)
            physical_cases=[dict(case_id=r['case_id'],source_id=r['source_id'],family=r['family'],
                endpoint_error_m=r.get('endpoint_error_m'),heading_error_rad=r.get('heading_error_rad'),
                endpoint_time_error_s=r.get('endpoint_time_error_s'),remaining_failures=[k for k in
                ('position','heading','timing','tracking_mean','tracking_p95','construction_complete','replay_complete') if not f[k]])
                for r,f in zip(rows,components) if f['physical']]
            near_misses={key:sum(all(f[k] for k in ('position','heading','timing',
                'surface_slip_proxy','penetration','continuity','tracking_mean',
                'tracking_p95','construction_complete','replay_complete') if k!=key) and not f[key]
                for f in components) for key in ('position','heading','timing','surface_slip_proxy','continuity')}
            entries.append(dict(arm=arm,seed=seed,attempts=144,component_pass_counts=passes,
                failure_signatures=dict(failures),physical_pass_cases=physical_cases,
                complete_construction_and_replay_physical_pass=sum(f['physical'] and f['construction_complete'] and f['replay_complete'] for f in components),
                physical_pass_but_construction_failed=sum(f['physical'] and not f['construction_complete'] for f in components),
                single_component_near_misses=near_misses))
            eligible={(r['source_id'],r['case_id']) for r,f in zip(rows,components) if f['physical']}
            eligibility[(arm,seed)]=eligible
            masks[(arm,seed)]={(r['source_id'],r['case_id']):f for r,f in zip(rows,components)}
            bindings[arm+':'+str(seed)]=dict(outputs=lineage['outputs'],selection_sha256=sha(selected))
    overlaps=[]
    for seed in (11,12,13):
        key=('proposed',seed)
        if key not in masks: continue
        for baseline in ('ordinary','random_best','native_best','no_feedback','shuffled','planned_prefix','tracking_only'):
            if (baseline,seed) not in masks: continue
            p=masks[key];b=masks[(baseline,seed)];assert p.keys()==b.keys()
            overlaps.append(dict(seed=seed,baseline=baseline,
                condition_repaired=sum(p[c]['condition'] and not b[c]['condition'] for c in p),
                condition_regressed=sum(b[c]['condition'] and not p[c]['condition'] for c in p),
                physical_repaired=sum(p[c]['physical'] and not b[c]['physical'] for c in p),
                physical_regressed=sum(b[c]['physical'] and not p[c]['physical'] for c in p),
                physically_passed_same_cases=eligibility[key]==eligibility[(baseline,seed)],
                joint_repaired=sum(p[c]['joint'] and not b[c]['joint'] for c in p)))
    construction=[]
    for path in sorted((BASE/'raw').iterdir()):
        if not path.name.startswith('test_') or path.name.startswith('test_eval_'):continue
        if not (path/'receipt.json').exists() or not (path/'lineage.json').exists():continue
        receipt=read(path/'receipt.json');lineage=read(path/'lineage.json');meta=read(path/'metadata.json')
        assert receipt['status']=='complete' and receipt['cases']==144 and meta['mode']!='replay'
        assert all(sha(path/name)==digest for name,digest in lineage['outputs'].items())
        rows=read(path/'metrics.json');assert len(rows)==144
        parts=[flags(r,dict(construction_complete=r['completed']),thresholds) for r in rows]
        construction.append(dict(label=path.name,seed=meta['seed'],round=meta['round'],mode=meta['mode'],
            attempts=len(rows),completed=sum(r['completed'] for r in rows),
            physical_pass=sum(f['physical'] for f in parts),
            completed_physical_pass=sum(f['physical'] and f['construction_complete'] for f in parts),
            joint_pass=sum(f['joint'] for f in parts),
            component_pass_counts={key:sum(f[key] for f in parts) for key in parts[0]},
            output_bindings=lineage['outputs']))
    payload=dict(time=time.time(),entries=entries,paired_component_changes=overlaps,
        construction_batches=construction,
        construction_and_replay_not_pooled=True,
        completed_arm_seed_batches=len(entries),expected_arm_seed_batches=30,
        all_replays_finished=len(entries)==30,all_observed_joint_zero=all(e['component_pass_counts']['joint']==0 for e in entries),
        thresholds_sha256=sha(BASE/'frozen_thresholds.json'),input_bindings=bindings,
        source_sha256=sha(Path(__file__)),live_pipeline_mutated=False,thresholds_changed=False,
        simulation_calls=0,generation_calls=0,innovation_confirmed=False,
        scope='Observed frozen-predicate component attribution, not causal or controller-independent diagnosis',
        next_action='Finish paired seeds; if final joint floor persists, retain negative evidence and diagnose before C2/G1/downstream')
    OUT.mkdir(exist_ok=True)
    destination=OUT/('diagnostic_'+str(time.time_ns())+'.json')
    destination.write_text(json.dumps(payload,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('entries','input_bindings','construction_batches')},indent=2))
    print('COMPONENT_DIAGNOSTIC_PATH',destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-check',action='store_true')
    main(parser.parse_args().self_check)

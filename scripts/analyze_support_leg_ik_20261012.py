"""Frozen raw-array analysis for classical positional IK diagnosis, not efficacy."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from prepare_support_leg_ik_20261012 import ROOT,PARENT,OUT
from verify_execution_lineage_v39_20261012 import read,sha,guard_batch
from physics_guided_metric_v23_20261012 import metrics


def main(freeze):
    lock=OUT/'analysis_plan_lock.json'
    if freeze:
        if lock.exists() or any((OUT/'raw'/('val_'+a+'_s11')).exists() for a in ('cartesian','two_link_ik')):
            raise RuntimeError('Freeze only once before either new simulator arm')
        lock.write_text(json.dumps(dict(source_sha256=sha(Path(__file__)),time=time.time(),
            purpose='Raw-array recomputation; paired source bootstrap; component and joint gains; no novelty acceptance'),indent=2),encoding='utf8')
        print('SUPPORT_LEG_IK_ANALYSIS_FROZEN',flush=True)
        return
    if not __debug__ or np.__version__!='1.23.5' or read(lock)['source_sha256']!=sha(Path(__file__)):
        raise RuntimeError('Frozen analysis source/runtime mismatch')
    summary=read(OUT/'summary.json')
    if summary['status']!='COMPLETE_LIMITED_LEG_IK_DIAGNOSIS':
        raise RuntimeError('Cannot analyze missing/partial outcomes as complete')
    bindings={}
    def bind(path):
        bindings[str(path.relative_to(ROOT))]=sha(path)
    version=read(OUT/'versions.json')
    for name,digest in version['files'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Bound code changed: '+name)
    for name in ('summary.json','preparation.json','versions.json','source_manifest.json','frozen_thresholds.json',
                 'DOCUMENT_WITNESS.json','preflight.json','analysis_plan_lock.json','launch_receipt.json',
                 'pipeline_finally_receipt.json','budget_ledger.json'):
        bind(OUT/name)
    for name in ('budget_ledger.json','numerical_adjudication_budget_ledger.json','NUMERICAL_CONTRACT_ADDENDUM.json'):
        bind(PARENT/name)
    limits=read(OUT/'frozen_thresholds.json')['thresholds']
    numeric=('endpoint_error_m','heading_error_rad','endpoint_time_error_s','surface_slip_proxy_fraction',
             'penetration_fraction','jerk_p95_m_s3','tracking_mean_m','tracking_p95_m')
    reports={};all_rows={};baseline_initial=None;baseline_meta=None;reset_checks={}
    for arm in ('baseline','cartesian','two_link_ik'):
        home=PARENT if arm=='baseline' else OUT
        path=home/'raw'/('val_'+arm+'_s11');wanted=home/'requests'/(path.name+'.json')
        guard_batch(path,read(wanted))
        meta=read(path/'metadata.json');stored=read(path/'metrics.json')
        with np.load(path/'trajectories.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        recomputed=[metrics(arrays,i,meta,limits) for i in range(72)]
        if len(stored)!=72 or [r['case_id'] for r in stored]!=read(OUT/'preparation.json')['source_case_order']:
            raise RuntimeError('Incorrect retained denominator/order')
        if baseline_meta is None:baseline_meta=meta
        for field in ('original_goal','original_heading','cases'):
            if meta[field]!=baseline_meta[field]:raise RuntimeError('Cross-arm input difference: '+field)
        for old,new in zip(stored,recomputed):
            for key in ('case_id','source_id','completed','condition_pass','physical_pass','tracking_pass','joint_pass'):
                if old[key]!=new[key]:raise RuntimeError('Raw classification mismatch: '+key)
            for key in numeric:
                a,b=old.get(key),new.get(key)
                if (a is None)!=(b is None) or (a is not None and not np.isclose(a,b,atol=1e-10,rtol=1e-10)):
                    raise RuntimeError('Raw metric mismatch: '+key)
        with np.load(path/'initial_states.npz',allow_pickle=False) as z:initial={k:z[k] for k in z.files}
        fields={'_humanoid_root_states','_dof_pos','_dof_vel','_rigid_body_pos','_rigid_body_rot','_rigid_body_vel','_rigid_body_ang_vel'}
        cache={'_rigid_body_pos','_rigid_body_rot','_rigid_body_vel','_rigid_body_ang_vel'}
        if set(meta['manual_reset_state_sha256'])!=fields or set(meta['initial_rigid_cache_consistency'])!=cache:
            raise RuntimeError('Incomplete reset/cache evidence')
        if not meta['initial_reset_gate_pass'] or any(v!=0 for v in meta['initial_rigid_cache_consistency'].values()):
            raise RuntimeError('Reset/cache gate failed')
        if any(not np.array_equal(initial['bootstrap_'+k],initial['first_action_'+k]) for k in fields):
            raise RuntimeError('Reset-to-first-action mismatch')
        if baseline_initial is None:baseline_initial=initial
        if set(initial)!=set(baseline_initial) or any(not np.array_equal(initial[k],baseline_initial[k]) for k in initial):
            raise RuntimeError('Cross-arm initial-state mismatch')
        reset_checks[arm]='Exported arrays exact; four cache checks pass; hidden PhysX state not established'
        counts={k:sum(int(r[k]) for r in recomputed) for k in ('completed','condition_pass','physical_pass','tracking_pass','joint_pass')}
        if counts!=summary['reports'][arm]['counts']:
            raise RuntimeError('Summary counts differ from raw arrays')
        finite={k:[r[k] for r in recomputed if r.get(k) is not None and np.isfinite(r[k])] for k in numeric}
        reports[arm]=dict(inputs=72,counts=counts,means={k:float(np.mean(v)) for k,v in finite.items() if v},
            metric_denominators={k:dict(finite=len(v),missing_or_nonfinite=72-len(v),total=72) for k,v in finite.items()})
        all_rows[arm]=recomputed
        for name in ('metadata.json','metrics.json','receipt.json','lineage.json','trajectories.npz','initial_states.npz','observation_trace.npz'):
            bind(path/name)
        bind(wanted);bind(OUT/'references'/(arm+'.npz'))
    sources=sorted({r['source_id'] for r in all_rows['baseline']})
    if len(sources)!=24:raise RuntimeError('Incorrect source grouping')
    paired={}
    for arm in ('cartesian','two_link_ik'):
        delta=[]
        for s in sources:
            ids=[i for i,r in enumerate(all_rows['baseline']) if r['source_id']==s]
            if len(ids)!=3:raise RuntimeError('Incorrect per-source offset count')
            delta.append(np.mean([int(all_rows[arm][i]['joint_pass'])-int(all_rows['baseline'][i]['joint_pass']) for i in ids]))
        delta=np.asarray(delta)
        boot=np.random.RandomState(6211).choice(delta,size=(5000,24)).mean(axis=1)*100
        paired[arm]=dict(joint_gain_pp=float(delta.mean()*100),source_cluster_95ci_pp=np.quantile(boot,[.025,.975]).tolist(),sources=24,resamples=5000,
            note='One-seed internal-validation diagnosis; a degenerate zero interval does not prove population equivalence')
    physical_gain=any(reports[a]['counts']['physical_pass']>reports['baseline']['counts']['physical_pass'] for a in ('cartesian','two_link_ik'))
    joint_gain=any(paired[a]['joint_gain_pp']>0 for a in paired)
    decision=('EXPLORATORY_JOINT_GAIN_NOT_INNOVATION_ACCEPTANCE' if joint_gain else
              'PHYSICAL_COMPONENT_GAIN_NO_JOINT_EFFICACY' if physical_gain else
              'STOP_THIS_LIMITED_LEG_TRANSLATION_AND_POSITION_IK_PARAMETERIZATION')
    result=dict(status='COMPLETE_RAW_RECOMPUTED_LEG_IK_DIAGNOSIS',reports=reports,paired=paired,decision=decision,
        reset_checks=reset_checks,input_hashes=bindings,source_sha256=sha(Path(__file__)),analysis_cpu_only=True,
        study_new_simulator_batches=2,analysis_new_simulator_batches=0,study_generation_calls=0,
        candidate_efficacy_test=False,confirmed_innovation=False,overleaf_results_sync=False,
        scope='Classical bounded positional IK; internal val with upstream train/test mix; no G1, downstream training, independent semantics/diversity or novelty acceptance')
    target=OUT/('analysis_%d.json'%time.time_ns());target.write_text(json.dumps(result,indent=2),encoding='utf8')
    print('SUPPORT_LEG_IK_ANALYSIS_COMPLETE',target,flush=True)
    print(json.dumps(dict(counts={a:r['counts'] for a,r in reports.items()},decision=decision),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');main(p.parse_args().freeze)

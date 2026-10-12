"""Read-only source-paired analysis; diagnosis, not an acceptance or novelty gate."""
from pathlib import Path
import json
import time
import numpy as np
from verify_execution_lineage_v39_20261012 import read,sha,guard_batch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/support_reference_probe_v63_20261012'


def main():
    assert __debug__ and np.__version__=='1.23.5'
    plan=read(OUT/'analysis_plan_lock.json')
    assert plan['analysis_source_sha256']==sha(Path(__file__))
    summary=read(OUT/'summary.json');assert summary['status']=='COMPLETE_REFERENCE_SIDE_DIAGNOSIS'
    version=read(OUT/'versions.json');assert all(sha(ROOT/n)==h for n,h in version['files'].items())
    inputs={str((OUT/'summary.json').relative_to(ROOT)):sha(OUT/'summary.json')}
    inputs[str((OUT/'versions.json').relative_to(ROOT))]=sha(OUT/'versions.json')
    inputs[str((OUT/'analysis_plan_lock.json').relative_to(ROOT))]=sha(OUT/'analysis_plan_lock.json')
    rows={};reports={};baseline_initial=None;reset_checks={}
    for arm in ('baseline','unconstrained','endpoint_preserved'):
        path=OUT/'raw'/('val_'+arm+'_s11');wanted=OUT/'requests'/(path.name+'.json')
        guard_batch(path,read(wanted))
        r=read(path/'metrics.json');meta=read(path/'metadata.json')
        assert len(r)==72 and [x['case_id'] for x in r]==read(OUT/'preparation.json')['source_case_order']
        assert meta['original_goal_sha256']==read(OUT/'preparation.json')['source_original_goal_sha256']
        initial_file=path/'initial_states.npz'
        assert sha(initial_file)==meta['initial_state_archive_sha256']
        with np.load(initial_file,allow_pickle=False) as archive:
            initial={key:archive[key] for key in archive.files}
        fields=tuple(meta['manual_reset_state_sha256'])
        assert len(fields)==7
        assert all(np.array_equal(initial['bootstrap_'+key],initial['first_action_'+key]) for key in fields)
        assert meta['initial_reset_gate_pass'] and all(x==0 for x in meta['initial_rigid_cache_consistency'].values())
        if baseline_initial is None:baseline_initial=initial
        assert set(initial)==set(baseline_initial)
        assert all(np.array_equal(initial[key],baseline_initial[key]) for key in initial)
        reset_checks[arm]=dict(exported_initial_states_exact_to_baseline=True,reset_cache_gate=True,
            scope='Exported kinematic tensors and cache consistency; does not establish hidden PhysX-state equality')
        rows[arm]=r
        numeric=('endpoint_error_m','heading_error_rad','endpoint_time_error_s','surface_slip_proxy_fraction',
                 'penetration_fraction','jerk_p95_m_s3','tracking_mean_m')
        values={key:[x[key] for x in r if isinstance(x.get(key),(int,float)) and np.isfinite(x[key])] for key in numeric}
        reports[arm]=dict(inputs=72,counts=summary['reports'][arm]['counts'],
            available_metric_means={key:float(np.mean(v)) for key,v in values.items() if v},
            metric_denominators={key:dict(finite=len(v),missing_or_nonfinite=72-len(v),total=72) for key,v in values.items()},
            evaluation_type='simulation_only; synthetic_proxy reference fidelity')
        for name in ('metadata.json','metrics.json','lineage.json','trajectories.npz','observation_trace.npz','initial_states.npz'):
            inputs[str((path/name).relative_to(ROOT))]=sha(path/name)
        inputs[str(wanted.relative_to(ROOT))]=sha(wanted)
    sources=sorted({r['source_id'] for r in rows['baseline']});assert len(sources)==24
    pairs={}
    for arm in ('unconstrained','endpoint_preserved'):
        changes=[]
        for source in sources:
            indices=[i for i,r in enumerate(rows['baseline']) if r['source_id']==source]
            assert len(indices)==3
            changes.append(float(np.mean([int(rows[arm][i]['joint_pass'])-int(rows['baseline'][i]['joint_pass']) for i in indices])))
        changes=np.asarray(changes)
        bootstrap=np.random.RandomState(6211).choice(changes,size=(5000,len(sources))).mean(axis=1)
        pairs[arm]=dict(source_count=24,joint_gain_pp=float(changes.mean()*100),
            source_cluster_95ci_pp=(np.quantile(bootstrap,[.025,.975])*100).tolist(),bootstrap_resamples=5000,
            note='One validation seed, exploratory; an all-zero interval does not prove population equivalence')
    joint_improved=any(r['joint_gain_pp']>0 for r in pairs.values())
    physical_improved=any(reports[a]['counts']['physical_pass']>reports['baseline']['counts']['physical_pass']
                          for a in ('unconstrained','endpoint_preserved'))
    if joint_improved:
        decision='EXPLORATORY_JOINT_GAIN_REQUIRES_STRONGER_BASELINES_SEMANTICS_THREE_SEEDS_AND_HELD_OUT_TEST'
    elif physical_improved:
        decision='PHYSICAL_ONLY_GAIN_NO_JOINT_EFFICACY_REQUIRES_FAILURE_ANALYSIS'
    else:
        decision='STOP_THIS_ROOT_TRANSLATION_PARAMETERIZATION_NO_PHYSICAL_OR_JOINT_GAIN'
    result=dict(status='COMPLETE_REFERENCE_SIDE_PAIRED_DIAGNOSIS',reports=reports,paired=pairs,decision=decision,reset_checks=reset_checks,
        inputs_per_arm=72,sources=24,seed=11,input_hashes=inputs,source_sha256=sha(Path(__file__)),
        generation_calls=0,simulation_calls=0,thresholds_changed=False,original_targets_changed=False,
        confirmed_innovation=False,overleaf_results_sync=False,
        scope='Three deterministic reference variants from one common cached construction; diagnostic only. Common construction acquisition costs are shared, not free. Endpoint pose preservation is not executed endpoint accuracy. Original-source clustering; no test data, G1, training or semantics/diversity gate.',time=time.time())
    dest=OUT/('analysis_%d.json'%time.time_ns());dest.write_text(json.dumps(result,indent=2))
    print('SUPPORT_REFERENCE_V63_ANALYSIS_COMPLETE',dest,flush=True)
    print(json.dumps(dict(reports=reports,paired=pairs,decision=decision),indent=2))


if __name__=='__main__':main()

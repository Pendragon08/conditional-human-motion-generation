"""Read-only recovery analysis of terminal V46; do not edit or rerun it."""
import json
from pathlib import Path
import time
from verify_execution_lineage_v39_20261012 import sha,read,guard_batch
from compare_reference_observation_v42_20261012 import compare

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/native_reference_velocity_v46_20261012'
PRIOR=ROOT/'results/reference_velocity_v43_20261012'
OUT=ROOT/'results/native_velocity_summary_v47_20261012'

def main():
    version=read(BASE/'versions.json')
    assert all(sha(ROOT/name)==digest for name,digest in version['files'].items())
    native=BASE/'raw/val_native_s11';replay=BASE/'raw/val_native_replay_s11'
    nm=read(native/'metadata.json');pm=read(PRIOR/'raw/val_planned_s11/metadata.json')
    assert nm['generation_noise_sha256']==pm['generation_noise_sha256']
    assert nm['cases']==pm['cases'] and nm['original_goal']==pm['original_goal']
    requests={}
    for label,path in (('val_native_s11',native),('val_native_replay_s11',replay)):
        expected=read(BASE/'requests'/ (label+'.json'))
        requests[label]=guard_batch(path,expected)
    interface=compare(native,replay)
    thresholds=read(BASE/'frozen_thresholds.json')['thresholds']
    batches={}
    for label,path in (('native_construction',native),('native_fixed_reference',replay),('planned_fixed_reference',PRIOR/'raw/val_replay_s11')):
        meta=read(path/'metadata.json');rows=read(path/'metrics.json')
        assert meta['cases']==nm['cases'] and meta['original_goal']==nm['original_goal'] and len(rows)==72
        batches[label]=dict(attempts=72,
            **{key:sum(r.get(key,False) for r in rows) for key in ('completed','condition_pass','physical_pass','tracking_pass','joint_pass')},
            slip_proxy_pass=sum(r.get('surface_slip_proxy_fraction') is not None and r['surface_slip_proxy_fraction']<=thresholds['sliding_fraction'] for r in rows),
            continuity_pass=sum(r['jerk_p95_m_s3']<=thresholds['continuity_jerk_p95_m_s3'] for r in rows),
            output_hashes={name:sha(path/name) for name in ('metadata.json','metrics.json','trajectories.npz')})
    inputs=[BASE/'versions.json',BASE/'budget_ledger.json',BASE/'frozen_thresholds.json',PRIOR/'raw/val_planned_s11/metadata.json']
    result=dict(status='COMPLETE_READ_ONLY_NATIVE_CONTROL_SUMMARY',original_launcher_status='FAILED_KEYERROR_GENERATION_NOISE_IN_REPLAY_METADATA',
        original_failure_preserved=True,source_sha256=sha(Path(__file__)),
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        batches=batches,strict_lineages=requests,interface=interface,
        paired_construction_noise_sha256=nm['generation_noise_sha256'],
        generation_calls=0,simulation_calls=0,candidate_efficacy=False,innovation_confirmed=False,
        thresholds_changed=False,historical_scores_changed=False,overleaf_results_sync=False,
        scope='Read-only summaries of two completed batches after terminal launcher failure; not a successful rerun or formal candidate efficacy. Noise is verified using construction metadata, never invented for replay.')
    OUT.mkdir(exist_ok=True)
    path=OUT/ ('summary_%d.json'%time.time_ns())
    path.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(batches,indent=2));print('READ_ONLY_SUMMARY',path)

if __name__=='__main__':
    main()

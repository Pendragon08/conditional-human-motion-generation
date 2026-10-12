"""Native CLoSD validation control after the V43 reference velocity repair."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time
import run_reference_velocity_v43_20261012 as engine
from verify_execution_lineage_v39_20261012 import read,sha

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'results/reference_velocity_v43_20261012'
OUT=ROOT/'results/native_reference_velocity_v46_20261012'

def prepare():
    assert read(PRIOR/'DOCUMENT_WITNESS.json')['status']=='PASS'
    control=ROOT/'results/original_reference_velocity_v45_20261012'
    assert read(control/'DOCUMENT_WITNESS.json')['status']=='PASS'
    assert read(control/'summary.json')['generation_calls']==0
    version=read(PRIOR/'versions.json')
    assert all(sha(ROOT/name)==digest for name,digest in version['files'].items())
    for name in ('scripts/run_native_velocity_v46_20261012.py','docs/native_velocity_v46_20261012.md'):
        version['files'][name]=sha(ROOT/name)
    version.update(role='Native CLoSD fixed-reference quality diagnostic with repaired velocity interface',
        simulation_batches=2,candidate_efficacy=False,innovation_confirmed=False,
        prior_interface_witness_sha256=sha(PRIOR/'DOCUMENT_WITNESS.json'),
        original_reference_control_witness_sha256=sha(control/'DOCUMENT_WITNESS.json'))
    if OUT.exists():
        assert read(OUT/'versions.json')==version
        return
    OUT.mkdir()
    for name in ('source_manifest.json','frozen_thresholds.json'):
        shutil.copyfile(PRIOR/name,OUT/name)
    engine.save(OUT/'versions.json',version);engine.save(OUT/'budget_ledger.json',[])
    (OUT/'raw').mkdir();(OUT/'requests').mkdir()

def main():
    prepare();engine.OUT=OUT
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        native=engine.batch('val_native_s11','native')
        import numpy as np
        with np.load(native/'trajectories.npz',allow_pickle=False) as arrays:
            metadata=read(native/'metadata.json')
            if not (OUT/'raw_reference.npz').exists():
                np.savez_compressed(OUT/'raw_reference.npz',reference=arrays['reference'],
                    original_goal=np.asarray(metadata['original_goal'],np.float32))
        replay=engine.batch('val_native_replay_s11','replay',OUT/'raw_reference.npz')
        from compare_reference_observation_v42_20261012 import compare
        trace=compare(native,replay)
        engine.save(OUT/'interface_summary.json',trace)
        t=read(OUT/'frozen_thresholds.json')['thresholds']
        batches={}
        for label,path in (('native_construction',native),('native_fixed_reference',replay),
            ('planned_fixed_reference',PRIOR/'raw/val_replay_s11')):
            rows=read(path/'metrics.json');meta=read(path/'metadata.json')
            assert len(rows)==72 and meta['cases']==metadata['cases']
            assert meta['original_goal']==metadata['original_goal']
            if label!='native_fixed_reference':
                assert meta['generation_noise_sha256']==metadata['generation_noise_sha256']
            batches[label]=dict(attempts=72,
                **{key:sum(r.get(key,False) for r in rows) for key in ('completed','condition_pass','physical_pass','tracking_pass','joint_pass')},
                slip_proxy_pass=sum(r.get('surface_slip_proxy_fraction') is not None and r['surface_slip_proxy_fraction']<=t['sliding_fraction'] for r in rows),
                continuity_pass=sum(r['jerk_p95_m_s3']<=t['continuity_jerk_p95_m_s3'] for r in rows),
                output_hashes={name:sha(path/name) for name in ('metadata.json','metrics.json','trajectories.npz')})
        summary=dict(status='COMPLETE_NATIVE_VELOCITY_DIAGNOSTIC',batches=batches,
            candidate_efficacy=False,innovation_confirmed=False,thresholds_changed=False,
            historical_scores_changed=False,overleaf_results_sync=False,
            scope='One validation seed; same source/text/goal/noise. Native CLoSD is an existing baseline. No candidate mechanism is tested and no C2/C3, G1 or downstream efficacy gate is bypassed.')
        engine.save(OUT/'summary.json',summary)
        engine.save(OUT/'state.json',dict(status=summary['status'],time=time.time()))
        print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args();main()

"""Validation-only original dataset replay with the V43 velocity interface."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

import run_reference_velocity_v43_20261012 as engine
from verify_execution_lineage_v39_20261012 import sha, read

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/original_reference_velocity_v45_20261012'
PRIOR=ROOT/'results/reference_velocity_v43_20261012'
ORIGINAL=ROOT/'results/mocap_reference_control_v33_20261012/raw/val_mocap_replay_s11'

def prepare():
    witness=read(PRIOR/'DOCUMENT_WITNESS.json')
    assert witness['status']=='PASS' and witness['efficacy_inferred'] is False
    version=read(PRIOR/'versions.json')
    assert all(sha(ROOT/name)==digest for name,digest in version['files'].items())
    assert sha(ORIGINAL/'dataset_reference.npz')==read(ORIGINAL/'metadata.json')['dataset_reference_sha256']
    for name in ('scripts/run_original_reference_velocity_v45_20261012.py',
                 'docs/original_reference_velocity_v45_20261012.md'):
        version['files'][name]=sha(ROOT/name)
    version.update(role='Original dataset reference with V43 interface; validation calibration only',
        simulation_batches=1,generation_calls=0,candidate_efficacy=False,innovation_confirmed=False,
        original_reference_sha256=sha(ORIGINAL/'dataset_reference.npz'),
        original_metadata_sha256=sha(ORIGINAL/'metadata.json'),
        prior_interface_witness_sha256=sha(PRIOR/'DOCUMENT_WITNESS.json'))
    if OUT.exists():
        assert read(OUT/'versions.json')==version
        return
    OUT.mkdir()
    for name in ('source_manifest.json','frozen_thresholds.json'):
        shutil.copyfile(PRIOR/name,OUT/name)
    engine.save(OUT/'versions.json',version)
    engine.save(OUT/'budget_ledger.json',[])
    (OUT/'raw').mkdir();(OUT/'requests').mkdir()

def main():
    prepare();engine.OUT=OUT
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=engine.batch('val_original_replay_s11','replay',ORIGINAL/'dataset_reference.npz')
        import numpy as np
        meta=read(path/'metadata.json');old=read(ORIGINAL/'metadata.json')
        assert meta['generation_batch_calls']==0
        assert meta['cases']==old['cases'] and meta['original_goal']==old['original_goal']
        with np.load(path/'trajectories.npz',allow_pickle=False) as a,np.load(ORIGINAL/'dataset_reference.npz',allow_pickle=False) as b:
            canonical=(b['reference']-b['reference'][:,:,0:1,:])+b['reference'][:,:,0:1,:]
            assert np.array_equal(a['reference'][a['valid']],canonical[a['valid']])
        rows=read(path/'metrics.json');t=read(OUT/'frozen_thresholds.json')['thresholds']
        assert len(rows)==72
        summary=dict(status='COMPLETE_ORIGINAL_REFERENCE_VELOCITY_DIAGNOSTIC',attempts=72,
            generation_calls=0,innovation_confirmed=False,candidate_efficacy=False,
            historical_scores_changed=False,thresholds_changed=False,overleaf_results_sync=False,
            pass_counts={key:sum(r.get(key,False) for r in rows) for key in ('completed','condition_pass','physical_pass','tracking_pass','joint_pass')},
            component_pass_counts=dict(slip_proxy=sum(r.get('surface_slip_proxy_fraction') is not None and r['surface_slip_proxy_fraction']<=t['sliding_fraction'] for r in rows),
                penetration=sum(r['penetration_fraction']<=t['penetration_fraction'] for r in rows),
                continuity=sum(r['jerk_p95_m_s3']<=t['continuity_jerk_p95_m_s3'] for r in rows)),
            by_offset={str(offset):dict(attempts=sum(r['offset_m']==offset for r in rows),
                **{key:sum(r.get(key,False) for r in rows if r['offset_m']==offset) for key in ('completed','physical_pass','tracking_pass','joint_pass')}) for offset in (-.15,0.,.15)},
            output_hashes={name:sha(path/name) for name in ('metadata.json','metrics.json','trajectories.npz','observation_trace.npz')},
            scope='One fixed controller and seed, validation only. Original references are MoCap-derived, not certified executable ground truth; failure cannot distinguish controller limitation from proxy misspecification or source defects.')
        engine.save(OUT/'summary.json',summary)
        engine.save(OUT/'state.json',dict(status=summary['status'],time=time.time()))
        print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args();main()

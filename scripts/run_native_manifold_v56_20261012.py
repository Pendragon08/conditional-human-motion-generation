"""Validation-only passive contact diagnostic; no candidate efficacy or training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import run_reset_alignment_checks_v4 as runtime
from physics_guided_com_lineage_v24_20261012 import request
from verify_execution_lineage_v39_20261012 import read,sha,guard_batch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/paired_feedback_validation_v51_20261012'
OUT=ROOT/'results/native_manifold_v56_20261012'
DOC=ROOT/'docs/native_manifold_v56_20261012.md'

def save(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf8')

def prepare():
    assert np.__version__=='1.23.5'
    old=read(BASE/'versions.json')
    assert all(sha(ROOT/name)==digest for name,digest in old['files'].items())
    assert sha(BASE/'source_manifest.json')==old['source_manifest_sha256']
    assert sha(BASE/'frozen_thresholds.json')==old['thresholds_sha256']
    files=dict(old['files'])
    for name in ('scripts/native_manifold_collect_v56_20261012.py','scripts/run_native_manifold_v56_20261012.py',str(DOC.relative_to(ROOT))):files[name]=sha(ROOT/name)
    version=dict(files=files,role='Passive native-contact validation diagnosis; no new algorithm or scoring',
        source_manifest_sha256=sha(BASE/'source_manifest.json'),thresholds_sha256=sha(BASE/'frozen_thresholds.json'),
        parent_versions_sha256=sha(BASE/'versions.json'),generator_weights_sha256=old['generator_weights_sha256'],
        controller_weights_sha256=old['controller_weights_sha256'],split='val',seed=11,cases_per_batch=72,
        sources=24,expected_replay_batches=2,generation_calls=0,own_gpu_wall_cap_seconds=600,
        confirmatory=False,innovation_confirmed=False,overleaf_results_sync=False)
    if OUT.exists():
        assert read(OUT/'versions.json')==version
        assert sha(OUT/'source_manifest.json')==version['source_manifest_sha256']
        assert sha(OUT/'frozen_thresholds.json')==version['thresholds_sha256']
        return
    OUT.mkdir();(OUT/'raw').mkdir();(OUT/'requests').mkdir()
    for name in ('source_manifest.json','frozen_thresholds.json'):shutil.copyfile(BASE/name,OUT/name)
    save(OUT/'versions.json',version);save(OUT/'budget_ledger.json',[])

def batch(arm):
    label='val_%s_replay_s11'%arm
    baseline=BASE/'raw'/label
    guard_batch(baseline,read(BASE/'requests'/(label+'.json')))
    replay=BASE/'selected'/('%s_s11.npz'%arm)
    expected=request(ROOT,OUT,'val','replay',11,0,dict(noise_round=0,replay_file=replay))
    assert expected['replay_motion']==read(BASE/'requests'/(label+'.json'))['replay_motion']
    path=OUT/'raw'/label;wanted=OUT/'requests'/(label+'.json')
    assert not path.exists(), 'Prior attempt retained; no automatic rerun'
    previous=read(sorted((ROOT/'results/round_resource_status_20261012').glob('snapshot_*.json'))[-1])
    dirs=[ROOT/name for name in previous['directories']]
    if OUT not in dirs:dirs.append(OUT)
    rows=[row for d in dirs if (d/'budget_ledger.json').exists() for row in read(d/'budget_ledger.json')]
    own=read(OUT/'budget_ledger.json')
    seconds=min(300,600-sum(x['seconds'] for x in own),72*3600-sum(x['seconds'] for x in rows),4*3600-sum(x['seconds'] for x in rows if x['stage']=='measurement'))
    assert seconds>=60
    assert shutil.disk_usage('/data').free>=20*1024**3
    assert sum(p.stat().st_size for d in dirs for p in d.rglob('*') if p.is_file())+previous['archive_bytes']<15*1024**3
    runtime.OUT=OUT;resources=runtime.resource_guard();save(wanted,expected)
    command=[str(runtime.PYTHON),'-u',str(ROOT/'scripts/native_manifold_collect_v56_20261012.py'),
        '--manifest',str(OUT/'source_manifest.json'),'--split','val','--mode','replay','--seed','11','--round','0',
        '--noise-round','0','--replay-file',str(replay),'--output',str(path),
        '--thresholds',str(OUT/'frozen_thresholds.json'),'--request-file',str(wanted)]
    save(OUT/'state.json',dict(status='RUNNING',label=label,pid=os.getpid(),command=command,resources=resources,time=time.time()))
    started=time.monotonic();code=-1
    try:
        with (OUT/(label+'.log')).open('x') as log:
            code=subprocess.run(command,cwd=ROOT/'third_party/CLoSD',env=runtime.environment(),stdout=log,stderr=subprocess.STDOUT,timeout=seconds).returncode
    finally:
        own.append(dict(label=label,stage='measurement',seconds=time.monotonic()-started,exit_code=code,request_sha256=sha(wanted)))
        save(OUT/'budget_ledger.json',own)
    assert code==0,'Failed attempt retained'
    record=guard_batch(path,expected);meta=read(path/'metadata.json')
    assert sha(path/'native_contacts.json')==meta['native_contacts_sha256']
    native=read(path/'native_contacts.json')
    assert not native['errors'] and len(native['records'])>0
    assert meta['generation_batch_calls']==0 and meta['dense_manifold_recording'] is True
    exact={}
    for name in ('trajectories.npz','observation_trace.npz','initial_states.npz','sensor_diagnostic.npz'):
        with np.load(path/name,allow_pickle=False) as new,np.load(baseline/name,allow_pickle=False) as old:
            assert new.files==old.files
            for field in new.files:exact[name+':'+field]=bool(np.array_equal(new[field],old[field]))
    assert all(exact.values()),'Instrumentation changes trajectory/observations; stop diagnostic'
    assert read(path/'metrics.json')==read(baseline/'metrics.json')
    result=dict(status='PASS_PASSIVE_EXPORT_EQUALITY',arm=arm,cases=72,exact_arrays=exact,
        frozen_metrics_exactly_equal=True,native_records=len(native['records']),native_errors=0,
        native_contact_sha256=sha(path/'native_contacts.json'),baseline_record=guard_batch(baseline,read(BASE/'requests'/(label+'.json'))),
        current_record=record,source_sha256=sha(Path(__file__)),generation_calls=0,
        evaluation_type='simulation_only; synthetic_proxy reference tracking',innovation_confirmed=False)
    save(OUT/(arm+'_equality.json'),result)
    return result

def main(check):
    prepare()
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if check:
                result=batch('feedback');save(OUT/'runtime_witness.json',result)
                save(OUT/'state.json',dict(status='WITNESS_COMPLETE_PENDING_REVIEW'))
                print('NATIVE_MANIFOLD_V56_DOCUMENT_CHECK_PASS',flush=True)
            else:
                witness=read(OUT/'DOCUMENT_WITNESS.json')
                assert witness['status']=='PASS' and witness['runtime_witness_sha256']==sha(OUT/'runtime_witness.json')
                assert sha(ROOT/'scripts/run_native_manifold_v56_20261012.py')==read(OUT/'versions.json')['files']['scripts/run_native_manifold_v56_20261012.py']
                batch('random')
                save(OUT/'summary.json',dict(status='COMPLETE_PASSIVE_NATIVE_CONTACT_DIAGNOSIS',arms={arm:read(OUT/(arm+'_equality.json')) for arm in ('feedback','random')},sources=24,cases_per_arm=72,seed=11,efficacy_test=False,metrics_replaced=False,innovation_confirmed=False))
                save(OUT/'state.json',dict(status='COMPLETE_PASSIVE_NATIVE_CONTACT_DIAGNOSIS'))
                print('NATIVE_MANIFOLD_V56_COMPLETE',flush=True)
        except Exception as e:
            save(OUT/'state.json',dict(status='FAILED_RETAINED',error=repr(e),pid=os.getpid()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true');a=p.parse_args();main(a.check)

"""Three-arm validation causal probe. No generator/controller training or gates."""
import argparse
import fcntl
from pathlib import Path
import json
import os
import shutil
import subprocess
import time
import run_reset_alignment_checks_v4 as runtime
from prepare_support_reference_probe_v62_20261012 import ROOT,BASE,OUT
from verify_execution_lineage_v39_20261012 import read,sha,guard_batch
from physics_guided_com_lineage_v24_20261012 import request


def save(path,x):path.write_text(json.dumps(x,indent=2),encoding='utf8')


def validate():
    assert __debug__
    version=read(OUT/'versions.json');assert all(sha(ROOT/n)==h for n,h in version['files'].items())
    assert sha(OUT/'source_manifest.json')==version['source_manifest_sha256']
    assert sha(OUT/'frozen_thresholds.json')==version['thresholds_sha256']
    assert all(sha(OUT/'references'/(arm+'.npz'))==h for arm,h in read(OUT/'preparation.json')['reference_hashes'].items())


def batch(arm,stage):
    validate();label='val_'+arm+'_s11';path=OUT/'raw'/label
    assert not path.exists(),'No rerun, preserve each failed attempt'
    expected=request(ROOT,OUT,'val','replay',11,0,dict(noise_round=0,replay_file=OUT/'references'/(arm+'.npz')))
    wanted=OUT/'requests'/(label+'.json');save(wanted,expected)
    prior=read(sorted((ROOT/'results/round_resource_status_20261012').glob('snapshot_*.json'))[-1])
    dirs=[ROOT/n for n in prior['directories']]
    if OUT not in dirs:dirs.append(OUT)
    rows=[r for d in dirs if (d/'budget_ledger.json').exists() for r in read(d/'budget_ledger.json')]
    own=read(OUT/'budget_ledger.json')
    timeout=min(300,600-sum(r['seconds'] for r in own),72*3600-sum(r['seconds'] for r in rows),
        (4 if stage=='measurement' else 12)*3600-sum(r['seconds'] for r in rows if r['stage']==stage))
    assert timeout>=60 and shutil.disk_usage('/data').free>=20*1024**3
    assert sum(p.stat().st_size for d in dirs for p in d.rglob('*') if p.is_file())+prior['archive_bytes']<15*1024**3
    runtime.OUT=OUT;resources=runtime.resource_guard()
    cmd=[str(runtime.PYTHON),'-u',str(ROOT/'scripts/reference_velocity_collect_v43_20261012.py'),
         '--manifest',str(OUT/'source_manifest.json'),'--split','val','--mode','replay','--seed','11','--round','0',
         '--noise-round','0','--replay-file',str(OUT/'references'/(arm+'.npz')),'--output',str(path),
         '--thresholds',str(OUT/'frozen_thresholds.json'),'--request-file',str(wanted)]
    save(OUT/'state.json',dict(status='RUNNING',arm=arm,pid=os.getpid(),resources=resources,time=time.time()))
    started=time.monotonic();code=-1
    try:
        with (OUT/(label+'.log')).open('x') as log:
            code=subprocess.run(cmd,cwd=ROOT/'third_party/CLoSD',env=runtime.environment(),stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
    finally:
        own.append(dict(label=label,stage=stage,seconds=time.monotonic()-started,exit_code=code,request_sha256=sha(wanted)))
        save(OUT/'budget_ledger.json',own)
    assert code==0,'Failed invocation retained'
    binding=guard_batch(path,expected);meta=read(path/'metadata.json');scores=read(path/'metrics.json')
    assert meta['generation_batch_calls']==0 and len(scores)==72
    assert meta['original_goal_sha256']==read(OUT/'preparation.json')['source_original_goal_sha256']
    assert [r['case_id'] for r in scores]==read(OUT/'preparation.json')['source_case_order']
    return dict(arm=arm,binding=binding,counts={key:sum(bool(r[key]) for r in scores) for key in
        ('completed','condition_pass','physical_pass','tracking_pass','joint_pass')},cases=72)


def main(check):
    validate()
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if check:
                result=batch('baseline','measurement')
                save(OUT/'runtime_witness.json',dict(status='PASS',result=result,versions_sha256=sha(OUT/'versions.json')))
                save(OUT/'state.json',dict(status='WITNESS_COMPLETE_PENDING_REVIEW'))
                print('SUPPORT_REFERENCE_V62_DOCUMENT_CHECK_PASS',flush=True)
            else:
                witness=read(OUT/'DOCUMENT_WITNESS.json')
                assert witness['status']=='PASS' and witness['runtime_witness_sha256']==sha(OUT/'runtime_witness.json')
                reports={'baseline':read(OUT/'runtime_witness.json')['result']}
                for arm in ('unconstrained','endpoint_preserved'):reports[arm]=batch(arm,'comparison')
                save(OUT/'summary.json',dict(status='COMPLETE_REFERENCE_SIDE_DIAGNOSIS',reports=reports,
                    seed=11,sources=24,inputs_per_arm=72,generator_calls=0,simulator_batches=3,
                    thresholds_changed=False,original_targets_changed=False,innovation_confirmed=False,
                    confirmatory=False,overleaf_results_sync=False,g1_or_downstream_run=False))
                save(OUT/'state.json',dict(status='COMPLETE_REFERENCE_SIDE_DIAGNOSIS'))
                print('SUPPORT_REFERENCE_V62_COMPLETE',flush=True)
        except Exception as e:
            save(OUT/'state.json',dict(status='FAILED_RETAINED',error=repr(e),pid=os.getpid()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true');a=p.parse_args();main(a.check)

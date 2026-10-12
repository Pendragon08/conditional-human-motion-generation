"""Two unused reference arms; baseline replay is explicitly reused and charged.

Frozen parent measurement interfaces; no training or innovation acceptance.
"""
import argparse
import fcntl
import json
import os
import time
import shutil
from pathlib import Path
import numpy as np
import run_support_reference_probe_v63_20261012 as probe
from prepare_support_leg_ik_20261012 import ROOT,BASE,PARENT,OUT
from support_leg_ik_core_20261012 import numeric_checks

DOC=ROOT/'docs/support_leg_ik_20261012.md'
probe.OUT=OUT


def verify():
    if not __debug__:
        raise RuntimeError('Optimized Python is not supported by inherited validators')
    probe.validate()
    parent=probe.read(PARENT/'NUMERICAL_CONTRACT_ADDENDUM.json')
    if parent['status']!='PASS' or parent['versions_sha256']!=probe.sha(PARENT/'versions.json'):
        raise RuntimeError('Original cached numerical adjudication changed')
    baseline=PARENT/'raw/val_baseline_s11'
    current=probe.guard_batch(baseline,probe.read(PARENT/'requests/val_baseline_s11.json'))
    prepared=probe.read(OUT/'preparation.json')
    if current!=prepared['baseline_binding']:
        raise RuntimeError('Cached baseline binding changed')
    if probe.sha(OUT/'references/baseline.npz')!=probe.sha(PARENT/'references/baseline.npz'):
        raise RuntimeError('Common cached baseline must be byte-identical')
    with np.load(OUT/'references/baseline.npz',allow_pickle=False) as z:
        base=z['reference'];goals=z['original_goal']
    for arm in ('cartesian','two_link_ik'):
        with np.load(OUT/'references'/(arm+'.npz'),allow_pickle=False) as z:
            x=z['reference'];g=z['original_goal']
        if x.shape!=base.shape or not np.isfinite(x).all() or not np.array_equal(goals,g):
            raise RuntimeError('Malformed reference or changed goal')
        if not np.array_equal(base[:61],x[:61]) or not np.array_equal(base[-1],x[-1]):
            raise RuntimeError('Prefix/endpose changed')
    return dict(baseline_binding=current,baseline_request_sha256=probe.sha(PARENT/'requests/val_baseline_s11.json'),
                parent_adjudication_sha256=probe.sha(PARENT/'NUMERICAL_CONTRACT_ADDENDUM.json'),numeric=numeric_checks())


def resource_preflight():
    prior=probe.read(sorted((ROOT/'results/round_resource_status_20261012').glob('snapshot_*.json'))[-1])
    own=probe.read(OUT/'budget_ledger.json')
    own_seconds=sum(r['seconds'] for r in own)
    # Reserve the whole child timeout, using the collector which includes the
    # separately charged CUDA adjudication rather than inherited ledger sums.
    if own_seconds+300>600 or prior['charged_wall_hours_one_gpu_per_job']*3600+own_seconds+300>72*3600:
        raise RuntimeError('Full-timeout global/own budget reservation failed')
    if prior['stage_hours']['comparison']*3600+own_seconds+300>12*3600:
        raise RuntimeError('Comparison stage reservation failed')
    if shutil.disk_usage('/data').free<20*1024**3:
        raise RuntimeError('Data free space below floor')
    dirs=[ROOT/n for n in prior['directories']]
    if OUT not in dirs:dirs.append(OUT)
    if sum(p.stat().st_size for d in dirs for p in d.rglob('*') if p.is_file())+prior['archive_bytes']>=15*1024**3:
        raise RuntimeError('Output cap exceeded')
    return prior


def main(check):
    evidence=verify()
    if check:
        if (OUT/'preflight.json').exists():
            raise RuntimeError('Retain first preflight; no automatic retry')
        probe.save(OUT/'preflight.json',dict(status='PASS',evidence=evidence,source_sha256=probe.sha(Path(__file__)),
            document_sha256=probe.sha(DOC),generation_calls=0,simulation_calls=0,time=time.time()))
        print('SUPPORT_LEG_IK_DOCUMENT_PREFLIGHT_PASS',flush=True)
        return
    follower=probe.read(OUT/'DOCUMENT_WITNESS.json')
    if follower['status']!='PASS' or follower['preflight_sha256']!=probe.sha(OUT/'preflight.json') or follower['source_sha256']!=probe.sha(Path(__file__)) or follower['document_sha256']!=probe.sha(DOC):
        raise RuntimeError('Requires bound independent document-following PASS')
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (OUT/'launch_receipt.json').open('x') as f:
            json.dump(dict(status='LAUNCHED',pid=os.getpid(),time=time.time(),source_sha256=probe.sha(Path(__file__)),
                versions_sha256=probe.sha(OUT/'versions.json'),witness_sha256=probe.sha(OUT/'DOCUMENT_WITNESS.json')),f,indent=2)
        code=1
        try:
            baseline=probe.read(PARENT/'runtime_witness.json')['result']
            reports={'baseline':dict(baseline,cached=True,cached_from=str(PARENT.relative_to(ROOT)),new_simulator_batches=0)}
            for arm in ('cartesian','two_link_ik'):
                verify();resource_preflight()
                reports[arm]=probe.batch(arm,'comparison')
            probe.save(OUT/'summary.json',dict(status='COMPLETE_LIMITED_LEG_IK_DIAGNOSIS',reports=reports,
                sources=24,seed=11,inputs_per_arm=72,new_simulator_batches=2,cached_baseline_batches=1,
                original_baseline_budget_ledger_sha256=probe.sha(PARENT/'budget_ledger.json'),
                thresholds_changed=False,original_targets_changed=False,generation_calls=0,
                innovation_confirmed=False,candidate_efficacy_test=False,overleaf_results_sync=False,
                limits='Classical position IK; no joint-angle limits, torque, collision or full dynamics guarantee'))
            probe.save(OUT/'state.json',dict(status='COMPLETE_LIMITED_LEG_IK_DIAGNOSIS'))
            code=0
            print('SUPPORT_LEG_IK_COMPLETE',flush=True)
        except Exception as error:
            probe.save(OUT/'state.json',dict(status='FAILED_RETAINED',error=repr(error),pid=os.getpid()))
            raise
        finally:
            with (OUT/'pipeline_finally_receipt.json').open('x') as f:
                json.dump(dict(phase='pipeline_finally',pipeline_code=code,pid=os.getpid(),time=time.time(),
                    note='Written by pipeline finally, not an external OS parent-exit observation'),f,indent=2)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true')
    main(p.parse_args().check)

"""Separate continuation after independent documented-invariant adjudication.

Preserves original FAIL and frozen preparation, references and analysis.
"""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path
import run_support_leg_ik_20261012 as original

probe=original.probe
ROOT=original.ROOT
OUT=original.OUT
PARENT=original.PARENT
DOC=ROOT/'docs/support_leg_ik_continuation_20261012.md'


def verify():
    evidence=original.verify()
    diagnosis=probe.read(OUT/'SCALAR_CONTRACT_DIAGNOSIS.json')
    prior=probe.read(OUT/'DOCUMENT_WITNESS.json')
    if diagnosis['status']!='PASS_DOCUMENTED_PREPARATION_INVARIANTS' or prior['status']!='FAIL':
        raise RuntimeError('Require retained original FAIL and separate independent adjudication')
    if not all(diagnosis['checks'].values()):
        raise RuntimeError('A documented preparation invariant remains unresolved')
    for name,digest in diagnosis['bindings']['original_files'].items():
        if probe.sha(OUT/name)!=digest:raise RuntimeError('Historical evidence changed: '+name)
    for arm,digest in diagnosis['bindings']['references'].items():
        if probe.sha(OUT/'references'/(arm+'.npz'))!=digest:
            raise RuntimeError('Adjudicated reference changed: '+arm)
    if diagnosis['bindings']['document_sha256']!=probe.sha(original.DOC):
        raise RuntimeError('Original document changed')
    if diagnosis['bindings']['preparation_source_sha256']!=probe.sha(ROOT/'scripts/prepare_support_leg_ik_20261012.py'):
        raise RuntimeError('Preparation source changed')
    if any(not f['check'].startswith('frozen_proxy_objective_records_') for f in prior['failures']):
        raise RuntimeError('Original FAIL includes unresolved non-scalar checks')
    plan=probe.read(OUT/'analysis_plan_lock.json')
    if plan['source_sha256']!=probe.sha(ROOT/'scripts/analyze_support_leg_ik_20261012.py'):
        raise RuntimeError('Pre-outcome analysis changed')
    files=['scripts/continue_support_leg_ik_20261012.py','docs/support_leg_ik_continuation_20261012.md',
           'scripts/analyze_support_leg_ik_20261012.py','results/support_leg_ik_20261012/SCALAR_CONTRACT_DIAGNOSIS.json',
           'results/support_leg_ik_20261012/DOCUMENT_WITNESS.json','results/support_leg_ik_20261012/preflight.json',
           'results/support_leg_ik_20261012/analysis_plan_lock.json']
    return dict(original_verification=evidence,files={n:probe.sha(ROOT/n) for n in files},
                original_witness_retained='FAIL',simulation_calls=0,generation_calls=0)


def main(check):
    evidence=verify()
    if check:
        with (OUT/'continuation_preflight.json').open('x') as f:
            json.dump(dict(status='PASS',evidence=evidence,time=time.time(),simulation_calls=0,
                source_sha256=probe.sha(Path(__file__)),document_sha256=probe.sha(DOC)),f,indent=2)
        print('SUPPORT_LEG_IK_CONTINUATION_PREFLIGHT_PASS',flush=True)
        return
    witness=probe.read(OUT/'CONTINUATION_DOCUMENT_WITNESS.json')
    if witness['status']!='PASS' or witness['preflight_sha256']!=probe.sha(OUT/'continuation_preflight.json') or witness['source_sha256']!=probe.sha(Path(__file__)) or witness['document_sha256']!=probe.sha(DOC):
        raise RuntimeError('Requires independently bound continuation document witness')
    before=probe.read(OUT/'continuation_preflight.json')['evidence']['files']
    if evidence['files']!=before:raise RuntimeError('Continuation inputs changed after preflight')
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (OUT/'launch_receipt.json').open('x') as f:
            json.dump(dict(status='LAUNCHED',pid=os.getpid(),time=time.time(),
                source_sha256=probe.sha(Path(__file__)),original_source_sha256=probe.sha(Path(original.__file__)),
                versions_sha256=probe.sha(OUT/'versions.json'),original_witness_sha256=probe.sha(OUT/'DOCUMENT_WITNESS.json'),
                continuation_witness_sha256=probe.sha(OUT/'CONTINUATION_DOCUMENT_WITNESS.json'),
                continuation_files=evidence['files']),f,indent=2)
        code=1
        try:
            baseline=probe.read(PARENT/'runtime_witness.json')['result']
            reports={'baseline':dict(baseline,cached=True,cached_from=str(PARENT.relative_to(ROOT)),new_simulator_batches=0)}
            for arm in ('cartesian','two_link_ik'):
                verify();original.resource_preflight()
                reports[arm]=probe.batch(arm,'comparison')
            probe.save(OUT/'summary.json',dict(status='COMPLETE_LIMITED_LEG_IK_DIAGNOSIS',reports=reports,
                sources=24,seed=11,inputs_per_arm=72,new_simulator_batches=2,cached_baseline_batches=1,
                original_baseline_budget_ledger_sha256=probe.sha(PARENT/'budget_ledger.json'),
                thresholds_changed=False,original_targets_changed=False,generation_calls=0,
                innovation_confirmed=False,candidate_efficacy_test=False,overleaf_results_sync=False,
                continuation_files=evidence['files'],original_witness_status='FAIL_RETAINED',
                limits='Classical position IK; no joint-angle limits, torque, collision or full dynamics guarantee'))
            probe.save(OUT/'state.json',dict(status='COMPLETE_LIMITED_LEG_IK_DIAGNOSIS'))
            code=0
            print('SUPPORT_LEG_IK_CONTINUATION_COMPLETE',flush=True)
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

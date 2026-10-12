"""Immutable third entry: explicit adjudications, unchanged experiment arms."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path
import continue_support_leg_ik_20261012 as prior

original=prior.original
probe=prior.probe
ROOT,OUT,PARENT=prior.ROOT,prior.OUT,prior.PARENT
DOC=ROOT/'docs/support_leg_ik_adjudicated_20261012.md'


def verify():
    evidence=prior.verify()
    amendment=probe.read(OUT/'CONTINUATION_CONTRACT_ADDENDUM.json')
    if amendment['status']!='PASS_DOCUMENTED_CONTINUATION_INVARIANTS' or not all(amendment['checks'].values()):
        raise RuntimeError('Actual documented continuation invariants must pass independently')
    expected={'original_witness_sha256':OUT/'CONTINUATION_DOCUMENT_WITNESS.json',
              'preflight_sha256':OUT/'continuation_preflight.json',
              'source_sha256':Path(prior.__file__),'document_sha256':prior.DOC}
    for key,path in expected.items():
        if amendment[key]!=probe.sha(path):raise RuntimeError('Separate adjudication binding mismatch: '+key)
    files=dict(evidence['files'])
    for path in (Path(__file__),DOC,OUT/'CONTINUATION_CONTRACT_ADDENDUM.json',OUT/'CONTINUATION_DOCUMENT_WITNESS.json',OUT/'continuation_preflight.json'):
        files[str(path.relative_to(ROOT))]=probe.sha(path)
    return dict(files=files,original_witness_status='FAIL_RETAINED',continuation_witness_status='FAIL_RETAINED',
                adjudication_status=amendment['status'],simulation_calls=0,generation_calls=0)


def main(check):
    evidence=verify()
    if check:
        with (OUT/'adjudicated_preflight.json').open('x') as f:
            json.dump(dict(status='PASS',evidence=evidence,source_sha256=probe.sha(Path(__file__)),
                document_sha256=probe.sha(DOC),time=time.time(),simulation_calls=0),f,indent=2)
        print('SUPPORT_LEG_IK_ADJUDICATED_PREFLIGHT_PASS',flush=True)
        return
    witness=probe.read(OUT/'ADJUDICATED_DOCUMENT_WITNESS.json')
    for key,path in {'preflight_sha256':OUT/'adjudicated_preflight.json','source_sha256':Path(__file__),'document_sha256':DOC}.items():
        if witness[key]!=probe.sha(path):raise RuntimeError('Independent witness binding mismatch: '+key)
    if witness['status']!='PASS' or evidence['files']!=probe.read(OUT/'adjudicated_preflight.json')['evidence']['files']:
        raise RuntimeError('Requires independently bound adjudicated preflight')
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (OUT/'launch_receipt.json').open('x') as f:
            json.dump(dict(status='LAUNCHED',pid=os.getpid(),time=time.time(),source_sha256=probe.sha(Path(__file__)),
                versions_sha256=probe.sha(OUT/'versions.json'),adjudicated_witness_sha256=probe.sha(OUT/'ADJUDICATED_DOCUMENT_WITNESS.json'),
                adjudicated_files=evidence['files'],historical_witnesses_retained=True),f,indent=2)
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
                adjudicated_files=evidence['files'],historical_witnesses_retained=True,
                limits='Classical position IK; no joint-angle limits, torque, collision or full dynamics guarantee'))
            probe.save(OUT/'state.json',dict(status='COMPLETE_LIMITED_LEG_IK_DIAGNOSIS'))
            code=0
            print('SUPPORT_LEG_IK_ADJUDICATED_COMPLETE',flush=True)
        except Exception as error:
            probe.save(OUT/'state.json',dict(status='FAILED_RETAINED',error=repr(error),pid=os.getpid()))
            raise
        finally:
            with (OUT/'pipeline_finally_receipt.json').open('x') as f:
                json.dump(dict(phase='pipeline_finally',pipeline_code=code,pid=os.getpid(),time=time.time(),
                    note='Pipeline finally receipt, not an external OS exit observation'),f,indent=2)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true')
    main(p.parse_args().check)

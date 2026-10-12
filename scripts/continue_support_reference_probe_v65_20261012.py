"""Continue two unused arms only after independent frozen-contract adjudication.

Original FAIL and invocation remain immutable. No simulation in --check.
"""
import argparse
import fcntl
import os
import time
from pathlib import Path
import run_support_reference_probe_v63_20261012 as probe

OUT=probe.OUT


def verify_freeze():
    probe.validate()
    failure=probe.read(OUT/'DOCUMENT_WITNESS.json')
    amendment=probe.read(OUT/'NUMERICAL_CONTRACT_ADDENDUM.json')
    assert failure['status']=='FAIL'
    assert amendment['status']=='PASS'
    assert amendment['original_witness_sha256']==probe.sha(OUT/'DOCUMENT_WITNESS.json')
    assert amendment['runtime_witness_sha256']==probe.sha(OUT/'runtime_witness.json')
    assert amendment['versions_sha256']==probe.sha(OUT/'versions.json')
    # Independent adjudication is necessary; launcher results alone are insufficient.
    assert amendment['simulation_rerun'] is False and amendment['source_changed'] is False
    assert amendment['experimental_thresholds_changed'] is False
    extra=probe.read(OUT/'numerical_adjudication_budget_ledger.json')
    seconds=sum(row['seconds'] for row in extra)
    assert seconds==amendment['charged_budget']['saved_array_cuda_seconds']
    assert all(row['exit_code']==0 and row['stage']=='measurement' and row['simulation_calls']==0 for row in extra)
    # Each unchanged batch has a maximum timeout of 300 s. Reserve that whole
    # timeout including the separate adjudication charge before launching it.
    assert sum(row['seconds'] for row in probe.read(OUT/'budget_ledger.json'))+seconds+300<=600
    bindings=failure['bindings']
    assert probe.sha(OUT/'versions.json')==bindings['versions_sha256']
    assert probe.sha(OUT/'preparation.json')==bindings['preparation_sha256']
    for arm,digest in bindings['reference_hashes'].items():
        assert probe.sha(OUT/'references'/(arm+'.npz'))==digest
    for name,digest in bindings['baseline']['output_hashes'].items():
        assert probe.sha(OUT/'raw/val_baseline_s11'/name)==digest
    assert probe.sha(OUT/'raw/val_baseline_s11/lineage.json')==bindings['baseline']['lineage_sha256']
    assert probe.sha(OUT/'raw/val_baseline_s11/receipt.json')==bindings['baseline']['receipt_sha256']
    return dict(original_witness_sha256=probe.sha(OUT/'DOCUMENT_WITNESS.json'),
                adjudication_sha256=probe.sha(OUT/'NUMERICAL_CONTRACT_ADDENDUM.json'),
                runtime_witness_sha256=probe.sha(OUT/'runtime_witness.json'),
                versions_sha256=probe.sha(OUT/'versions.json'),
                preparation_sha256=probe.sha(OUT/'preparation.json'),
                numerical_budget_sha256=probe.sha(OUT/'numerical_adjudication_budget_ledger.json'))


def verify():
    evidence=verify_freeze()
    assert not (OUT/'summary.json').exists()
    for arm in ('unconstrained','endpoint_preserved'):
        assert not (OUT/'raw'/('val_'+arm+'_s11')).exists()
    baseline=OUT/'raw/val_baseline_s11'
    probe.guard_batch(baseline,probe.read(OUT/'requests/val_baseline_s11.json'))
    return evidence


def main(check):
    evidence=verify()
    if check:
        probe.save(OUT/'continuation_v65_preflight.json',dict(status='PASS',evidence=evidence,
            source_sha256=probe.sha(Path(__file__)),simulation_calls=0,generation_calls=0,time=time.time()))
        print('SUPPORT_REFERENCE_V65_PREFLIGHT_PASS',flush=True)
        return
    follower=probe.read(OUT/'CONTINUATION_DOCUMENT_WITNESS_V65.json')
    assert follower['status']=='PASS'
    assert follower['source_sha256']==probe.sha(Path(__file__))
    assert follower['preflight_sha256']==probe.sha(OUT/'continuation_v65_preflight.json')
    assert follower['document_sha256']==probe.sha(probe.ROOT/'docs/support_reference_continuation_v65_20261012.md')
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (OUT/'continuation_v65_receipt.json').open('x') as handle:
            import json
            json.dump(dict(status='LAUNCHED',pid=os.getpid(),time=time.time(),evidence=evidence,
                source_sha256=probe.sha(Path(__file__)),follower_sha256=probe.sha(OUT/'CONTINUATION_DOCUMENT_WITNESS_V65.json')),
                handle,indent=2)
        reports={'baseline':probe.read(OUT/'runtime_witness.json')['result']}
        try:
            for arm in ('unconstrained','endpoint_preserved'):
                assert verify_freeze()==evidence
                reports[arm]=probe.batch(arm,'comparison')
            probe.save(OUT/'summary.json',dict(status='COMPLETE_REFERENCE_SIDE_DIAGNOSIS',reports=reports,
                seed=11,sources=24,inputs_per_arm=72,generator_calls=0,simulator_batches=3,
                thresholds_changed=False,original_targets_changed=False,innovation_confirmed=False,
                confirmatory=False,overleaf_results_sync=False,g1_or_downstream_run=False,
                original_document_verdict='FAIL_RETAINED',continuation_evidence=evidence))
            probe.save(OUT/'state.json',dict(status='COMPLETE_REFERENCE_SIDE_DIAGNOSIS'))
            print('SUPPORT_REFERENCE_V65_CONTINUATION_COMPLETE',flush=True)
        except Exception as exc:
            probe.save(OUT/'state.json',dict(status='FAILED_RETAINED',error=repr(exc),pid=os.getpid()))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--run',action='store_true')
    main(parser.parse_args().check)

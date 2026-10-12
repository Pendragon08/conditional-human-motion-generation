"""Explicit missing-batch continuation, with a CPU-only preflight.

Preserves V56/V58/V60 failures. Reuses V56 batch executor and validates actual
Path-based hash calls before spending further simulation time.
"""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
import run_native_manifold_v56_20261012 as core
from verify_execution_lineage_v39_20261012 import read,sha

def preflight():
    if sys.flags.optimize:raise RuntimeError('Assertions must be enabled')
    core.prepare();out=core.OUT
    assert not (out/'continuation_manifest_v61.json').exists(),'No automatic retry'
    assert not (out/'raw/val_random_replay_s11').exists(),'Missing batch only'
    first=read(out/'failed_main_invocation_state.json')
    assert first['status']=='FAILED_RETAINED' and first['error']=="FileNotFoundError(2, 'No such file or directory')"
    assert sha(out/'failed_main_invocation_state.json')=='656983246104825dbb7e7e845d73fd73479f7858c98f0a2e0df9fbbb1be5a8d9'
    for name in ('CONTINUATION_DOCUMENT_WITNESS_V58.json','CONTINUATION_DOCUMENT_WITNESS_V60.json'):
        failure=read(out/name)
        assert failure['exit_code']==1 and failure['status']=='FAILED_RETAINED'
        assert not failure['random_batch_started']
    witness=read(out/'DOCUMENT_WITNESS.json')
    assert witness['status']=='PASS_DOCUMENT_FOLLOWING_RUNTIME_WITNESS' and witness['exit_code']==0
    assert witness['invocation_count']==1 and witness['all_exact_arrays'] and witness['frozen_metrics_exactly_equal']
    assert witness['generation_calls']==0 and witness['parser_errors_zero'] and witness['native_contact_records_nonzero']
    assert witness['runtime_witness_sha256']==sha(out/'runtime_witness.json')
    assert witness['doc_sha256']==sha(core.DOC)
    assert witness['source_sha256']==sha(core.ROOT/'scripts/run_native_manifold_v56_20261012.py')
    assert len(read(out/'budget_ledger.json'))==1
    paths=[Path(__file__),core.ROOT/'scripts/run_native_manifold_v56_20261012.py',
        core.ROOT/'docs/native_manifold_continuation_v61_20261012.md',out/'DOCUMENT_WITNESS.json',
        out/'runtime_witness.json',out/'failed_main_invocation_state.json',
        out/'CONTINUATION_DOCUMENT_WITNESS_V58.json',out/'CONTINUATION_DOCUMENT_WITNESS_V60.json']
    return dict(status='PREFLIGHT_PASS',assertions_enabled=True,
        input_hashes={str(p.relative_to(core.ROOT)):sha(p) for p in paths},
        source_sha256=sha(Path(__file__)),generation_calls=0,simulation_calls=0,time=time.time())

def main(check):
    evidence=preflight();out=core.OUT
    with (out/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if check:
            p=out/('preflight_v61_%d.json'%time.time_ns());core.save(p,evidence)
            print('NATIVE_MANIFOLD_V61_PREFLIGHT_PASS',p,flush=True);return
        evidence.update(status='CONTINUATION_STARTED',pid=os.getpid(),command=[sys.executable]+sys.argv,
            preserved_prebatch_failures=['V56 witness copy','V58 error repr','V60 Path hash'],
            original_failed_invocations_not_success=True,innovation_confirmed=False,
            evaluation_reference_type='synthetic_proxy; prior model-generated physics-refined trajectory')
        core.save(out/'continuation_manifest_v61.json',evidence)
        try:
            core.batch('random')
            core.save(out/'summary.json',dict(status='COMPLETE_PASSIVE_NATIVE_CONTACT_DIAGNOSIS',
                arms={arm:read(out/(arm+'_equality.json')) for arm in ('feedback','random')},sources=24,
                cases_per_arm=72,seed=11,efficacy_test=False,metrics_replaced=False,innovation_confirmed=False,
                original_main_invocations_failed=True,continuation='V61',
                continuation_source_sha256=sha(Path(__file__)),continuation_manifest_sha256=sha(out/'continuation_manifest_v61.json')))
            core.save(out/'state.json',dict(status='COMPLETE_PASSIVE_NATIVE_CONTACT_DIAGNOSIS',continuation='V61'))
            core.save(out/'continuation_receipt_v61.json',dict(status='COMPLETE',exit_code=0,
                continuation_manifest_sha256=sha(out/'continuation_manifest_v61.json'),summary_sha256=sha(out/'summary.json'),
                budget_ledger_sha256=sha(out/'budget_ledger.json'),generation_calls=0,simulation_batches_this_invocation=1))
            print('NATIVE_MANIFOLD_V61_CONTINUATION_COMPLETE',flush=True)
        except Exception as e:
            core.save(out/'continuation_receipt_v61.json',dict(status='FAILED_RETAINED',error=repr(e),
                continuation_manifest_sha256=sha(out/'continuation_manifest_v61.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--preflight',action='store_true');g.add_argument('--run',action='store_true');a=p.parse_args();main(a.preflight)

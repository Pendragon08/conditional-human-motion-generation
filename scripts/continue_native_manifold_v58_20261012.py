"""Explicit continuation after pre-batch V56 witness-copy/gate failure.

No V56 source or failed-state rewrite; validates descriptive witness PASS enum,
actual document/source/runtime hashes, then runs only the missing random replay.
"""
import fcntl
import os
import sys
import time
import run_native_manifold_v56_20261012 as core
from verify_execution_lineage_v39_20261012 import read,sha

def main():
    if sys.flags.optimize:raise RuntimeError('Assertions must be enabled')
    core.prepare();out=core.OUT
    with (out/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (out/'continuation_manifest_v58.json').exists(),'Prior continuation attempt retained'
        assert not (out/'raw/val_random_replay_s11').exists(),'No duplicate replay'
        failure=read(out/'failed_main_invocation_state.json')
        assert failure['status']=='FAILED_RETAINED' and 'DOCUMENT_WITNESS.json' in failure['error']
        witness=read(out/'DOCUMENT_WITNESS.json')
        assert witness['status'] in ('PASS','PASS_DOCUMENT_FOLLOWING_RUNTIME_WITNESS')
        assert witness['exit_code']==0 and witness['invocation_count']==1
        assert witness['runtime_witness_sha256']==sha(out/'runtime_witness.json')
        assert witness['doc_sha256']==sha(core.DOC)
        assert witness['source_sha256']==sha(core.ROOT/'scripts/run_native_manifold_v56_20261012.py')
        assert witness['all_exact_arrays'] and witness['frozen_metrics_exactly_equal'] and witness['generation_calls']==0
        assert witness['parser_errors_zero'] and witness['native_contact_records_nonzero']
        assert len(read(out/'budget_ledger.json'))==1
        doc=core.ROOT/'docs/native_manifold_continuation_v58_20261012.md'
        binding=dict(status='CONTINUATION_STARTED',pid=os.getpid(),command=[sys.executable]+sys.argv,
            assertions_enabled=sys.flags.optimize==0,source_sha256=sha(__file__),doc_sha256=sha(doc),
            witness_sha256=sha(out/'DOCUMENT_WITNESS.json'),runtime_witness_sha256=sha(out/'runtime_witness.json'),
            failed_main_state_sha256=sha(out/'failed_main_invocation_state.json'),time=time.time(),
            original_v56_source_changed=False,original_failed_invocation_not_success=True,
            generation_calls=0,innovation_confirmed=False)
        core.save(out/'continuation_manifest_v58.json',binding)
        try:
            core.batch('random')
            core.save(out/'summary.json',dict(status='COMPLETE_PASSIVE_NATIVE_CONTACT_DIAGNOSIS',
                arms={arm:read(out/(arm+'_equality.json')) for arm in ('feedback','random')},sources=24,
                cases_per_arm=72,seed=11,efficacy_test=False,metrics_replaced=False,innovation_confirmed=False,
                original_main_invocation_failed=True,continuation_source_sha256=sha(__file__),
                continuation_manifest_sha256=sha(out/'continuation_manifest_v58.json')))
            core.save(out/'state.json',dict(status='COMPLETE_PASSIVE_NATIVE_CONTACT_DIAGNOSIS',continuation='V58'))
            core.save(out/'continuation_receipt_v58.json',dict(status='COMPLETE',exit_code=0,
                continuation_manifest_sha256=sha(out/'continuation_manifest_v58.json'),summary_sha256=sha(out/'summary.json'),
                budget_ledger_sha256=sha(out/'budget_ledger.json'),generation_calls=0,simulation_batches_this_invocation=1))
            print('NATIVE_MANIFOLD_V58_CONTINUATION_COMPLETE',flush=True)
        except Exception as e:
            core.save(out/'continuation_receipt_v58.json',dict(status='FAILED_RETAINED',error=repr(e),
                continuation_manifest_sha256=sha(out/'continuation_manifest_v58.json')));raise

if __name__=='__main__':main()

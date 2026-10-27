"""New-run adapter. Reuse unchanged CPU serializer and export saved statistics.

Positive loading/export is not yet witnessed until a REAL new checkpoint exists.
"""
import argparse
import json
import os
import pathlib
import sys
from filter_recovery_backup_protocol import RUN, SPEC, preflight, sha

ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--completed', type=int, required=True)
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args()
    assert os.getuid() == 1013
    os.sched_setaffinity(0, {0, 1, 2, 3})
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    sp = ROOT / SPEC; s = json.loads(sp.read_text(encoding='utf-8'))
    result = preflight(ROOT, s, a.completed)
    if a.preflight_only or not result['ready']:
        print(json.dumps(result)); return
    assert s['backup_enabled'] is True
    import fcntl
    guard = (ROOT / 'receipts/filter_recovery_checkpoint_audit.lock').open('a+')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from filter_recovery_destination import guard_destination
    destination = ROOT / 'checkpoint_backups' / ('round_' + str(a.completed)) / s['audit_run']
    output_guard = guard_destination(ROOT, destination)
    import inspect_formal_filter_checkpoint_v2 as core
    assert sha(core.__file__) == s['bound_backup_files']['scripts/inspect_formal_filter_checkpoint_v2.py']
    core.RUN = RUN
    sys.argv = [core.__file__, '--completed', str(a.completed), '--audit-run', s['audit_run']]
    core.main()
    guard_destination(ROOT, destination, require_absent=False)
    audit = json.loads((destination / 'audit.json').read_text(encoding='utf-8'))
    assert audit['run'] == RUN and audit['full_checkpoint']['sha256'] == result['metadata']['sha256']
    # Same trusted own checkpoint, CPU-only. Original serializer is not edited.
    import torch
    from filter_recovery_statistics import reference_rows, summarize
    assert not torch.cuda.is_initialized()
    cp = pathlib.Path(result['checkpoint']); assert sha(cp) == audit['full_checkpoint']['sha256']
    saved = torch.load(cp, map_location='cpu', weights_only=False)
    assert saved['infos']['completed_learning_iterations'] == a.completed
    assert saved['infos']['actual_optimizer_updates'] == a.completed * 20
    rows = reference_rows(ROOT, s['training_receipt_bindings']['reference_fingerprint'], sha)
    keys = ['raw_failed', 'raw_success', 'failed_motion_count', 'success_motion_count']
    counters = {}
    for key in keys:
        t = saved['infos']['recording'][key]
        assert tuple(t.shape) == (8278,) and t.device.type == 'cpu'
        counters[key] = t.tolist()
    summary, decisions = summarize(rows, counters)
    private = destination / 'relaxed_source_decisions.private.json'
    with private.open('x', encoding='utf-8') as f:
        json.dump(dict(schema='textop.filter_recovery_relaxed.v1', run=RUN, completed=a.completed,
            checkpoint_sha256=sha(cp), reference_fingerprint=s['training_receipt_bindings']['reference_fingerprint'],
            motions=decisions), f, indent=2, allow_nan=False)
    summary.update(status='RECOVERY_ALL_SOURCE_RELAXED_STATISTICS_EXPORTED_STRICT_PENDING', run=RUN,
        completed_learning_iterations=a.completed, actual_optimizer_updates=a.completed * 20, paper_reference_rounds=40000,
        checkpoint_sha256=sha(cp), reference_fingerprint=s['training_receipt_bindings']['reference_fingerprint'],
        private_decisions=dict(path=str(private), bytes=private.stat().st_size, sha256=sha(private)),
        statistics_source_sha256=sha(ROOT / 'scripts/filter_recovery_statistics.py'),
        limitation='Historical saved relaxed EMA includes training history. Candidates require separate full-length strict physics. Not absolute infeasibility or corpus acceptance.')
    with (destination / 'relaxed_summary.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    after = preflight(ROOT, s, a.completed)
    assert after['ready'] and after['metadata'] == result['metadata'] and not torch.cuda.is_initialized()
    proof = dict(status='FILTER_RECOVERY_CHECKPOINT_CPU_AUDIT_AND_RELAXED_EXPORT_COMPLETE', run=RUN, completed=a.completed,
        adapter_sha256=sha(__file__), protocol_sha256=sha(ROOT / 'scripts/filter_recovery_backup_protocol.py'),
        spec_sha256=sha(sp), preflight=after, audit_sha256=sha(destination / 'audit.json'),
        output_guard=output_guard,
        relaxed_summary_sha256=sha(destination / 'relaxed_summary.json'), sdk_runs=0, new_optimizer_updates=0,
        strict_physical_filter_completed=False, dataset_accepted=False)
    with (destination / 'adapter_proof.json').open('x', encoding='utf-8') as f:
        json.dump(proof, f, indent=2)


if __name__ == '__main__':
    main()

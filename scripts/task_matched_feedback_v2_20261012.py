"""Separate preparation after relative __file__ CPU failure; no failed GPU retry."""
import argparse
import json
from pathlib import Path
import fcntl
import task_matched_feedback_20261012 as run

ROOT = run.ROOT
OUT = ROOT / 'results/task_matched_feedback_v2_20261012'
run.OUT = OUT
run.prior.OUT = OUT
run.DOC = ROOT / 'docs/task_matched_feedback_v2_20261012.md'


def main(mode):
    failure = run.read(ROOT / 'results/task_matched_feedback_20261012/PREPARATION_FAILURE.json')
    if failure['actual_exit_code'] != 1 or failure['gpu_batches_started'] != 0:
        raise RuntimeError('Original failed preparation must be preserved')
    if mode == 'prepare':
        if OUT.exists(): raise RuntimeError('Never reprepare an existing attempt')
        # Import resolves the original module __file__ absolutely. Its frozen
        # source is unchanged; a separate attempt/entry binds this distinction.
        run.prepare()
        version = run.read(OUT / 'versions.json')
        version['files'][str(Path(__file__).resolve().relative_to(ROOT))] = run.sha(Path(__file__))
        version['original_preparation_failure_sha256'] = run.sha(ROOT / 'results/task_matched_feedback_20261012/PREPARATION_FAILURE.json')
        run.save(OUT / 'versions.json', version)
        run.validate()
        print('TASK_MATCHED_FEEDBACK_V2_PREPARED', flush=True)
        return
    run.validate()
    run.prior.prepare = run.validate
    run.prior.batch = run.batch
    if mode == 'check':
        with (OUT / 'queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run.check()
    else:
        witness = run.read(OUT / 'DOCUMENT_WITNESS.json')
        if witness['status'] != 'PASS' or witness['versions_sha256'] != run.sha(OUT / 'versions.json'):
            raise RuntimeError('Independent task-matched witness required')
        run.prior.main(False)
        print('TASK_MATCHED_FEEDBACK_VALIDATION_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); g = p.add_mutually_exclusive_group(required=True)
    for mode in ('prepare', 'check', 'run'): g.add_argument('--' + mode, action='store_true')
    a = p.parse_args()
    main(next(m for m in ('prepare', 'check', 'run') if getattr(a, m)))

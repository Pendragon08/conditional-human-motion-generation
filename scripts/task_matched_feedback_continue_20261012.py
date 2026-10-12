"""Prospective dispatcher allocation; scientific inputs/weights/thresholds fixed.

Old 1800s entry is not extended or rerun. Carry only its already witnessed
first candidate; new trials are in this separate directory under 3600s cap.
"""
import argparse
import inspect
import json
from pathlib import Path
import shutil
import fcntl
import task_matched_feedback_v2_20261012 as previous

run = previous.run
ROOT = run.ROOT
OLD = ROOT / 'results/task_matched_feedback_v2_20261012'
OUT = ROOT / 'results/task_matched_feedback_continue_20261012'
DOC = ROOT / 'docs/task_matched_feedback_continue_20261012.md'
run.OUT = OUT
run.prior.OUT = OUT
# The only alteration to the actual batch function is dispatcher wall budget;
# loaded data, controller, generator, targets, candidate counts and scores stay
# fixed. Record the exact assembled function in the new frozen version.
assembled = inspect.getsource(run.batch)
old = '1800 - sum(r[\'seconds\'] for r in ledger)'
if assembled.count(old) != 1: raise RuntimeError('Unexpected dispatcher source')
assembled = assembled.replace(old, '3600 - sum(r[\'seconds\'] for r in ledger)')
exec(compile(assembled, str(Path(__file__)), 'exec'), run.__dict__)
normal_batch = run.batch


def carried_first():
    v = run.read(OUT / 'versions.json')
    for name, digest in v['carried_first_hashes'].items():
        if run.sha(ROOT / name) != digest: raise RuntimeError('Carried first input changed')
    source = OLD / 'raw/val_shared_s11_r0'
    wanted = OLD / 'requests/val_shared_s11_r0.json'
    run.guard_batch(source, run.read(wanted))
    # Verify against its original execution versions, never relabel old request.
    expected, original = run.read(wanted), run.read(OLD / 'versions.json')
    meta = run.read(source / 'metadata.json')
    data = run.read(source / 'generation_input_binding.json')
    weights = run.read(source / 'loaded_checkpoint_binding.json')
    if data['exact_loaded_dataset_bytes'] != original['external_dataset_files']:
        raise RuntimeError('Carried dataset differs')
    if data['loaded_correction_object_sha256'] != expected.get('correction'):
        raise RuntimeError('Carried correction differs')
    if any(binding['request_sha256'] != run.sha(wanted) for binding in (data, weights)):
        raise RuntimeError('Carried request differs')
    if weights['versions_sha256'] != run.sha(OLD / 'versions.json') or weights['controller_state'] != run.STATE:
        raise RuntimeError('Carried execution version differs')
    if sorted(weights['actual_loaded_weight_bytes'].values()) != sorted([original['generator_weights_sha256'], original['controller_states'][run.STATE]['sha256']]):
        raise RuntimeError('Carried actual weights differ')
    if run.sha(source / 'initial_states.npz') != meta['initial_state_archive_sha256'] or run.sha(source / 'observation_trace.npz') != meta['observation_trace_sha256']:
        raise RuntimeError('Carried archives differ')
    if not meta['initial_reset_gate_pass'] or any(meta['initial_rigid_cache_consistency'].values()):
        raise RuntimeError('Carried reset inconsistent')
    return source


def batch(label, mode, seed, ordinal=0, correction=None, replay=None, stage='comparison'):
    if label == 'val_shared_s11_r0':
        if (mode, seed, ordinal, correction, replay) != ('native', 11, 0, None, None):
            raise RuntimeError('Cannot substitute carried candidate')
        return carried_first()
    return normal_batch(label, mode, seed, ordinal, correction, replay, stage)


def prepare():
    if OUT.exists(): raise RuntimeError('Never overwrite allocation attempt')
    old = run.read(OLD / 'versions.json')
    w = run.read(OLD / 'DOCUMENT_WITNESS.json')
    if w['status'] != 'PASS' or w['runtime_witness_sha256'] != run.sha(OLD / 'runtime_witness.json'):
        raise RuntimeError('Actual generation/correction/replay witness required')
    if w['versions_sha256'] != run.sha(OLD / 'versions.json'):
        raise RuntimeError('Original witnessed versions changed')
    OUT.mkdir(); (OUT / 'raw').mkdir(); (OUT / 'requests').mkdir()
    for name in ('source_manifest.json', 'frozen_thresholds.json'):
        shutil.copyfile(OLD / name, OUT / name)
    for path in (Path(__file__).resolve(), DOC, ROOT / 'docs/task_matched_feedback_20261012.md'):
        old['files'][str(path.relative_to(ROOT))] = run.sha(path)
    parent = OLD / 'raw/val_shared_s11_r0'
    names = [parent / n for n in ('metadata.json', 'metrics.json', 'lineage.json', 'receipt.json',
             'trajectories.npz', 'initial_states.npz', 'observation_trace.npz', 'generation_input_binding.json',
             'loaded_checkpoint_binding.json', 'loaded_input_binding.json')]
    names += [OLD / 'versions.json', OLD / 'DOCUMENT_WITNESS.json', OLD / 'runtime_witness.json',
              OLD / 'requests/val_shared_s11_r0.json']
    old.update(max_new_seconds=3600, scope='Same scientific C1 validation, separate prospective dispatch allocation',
               carried_first_hashes={str(p.relative_to(ROOT)):run.sha(p) for p in names},
               batch_implementation_sha256=run.object_hash(assembled),
               original_measurement_seconds=sum(r['seconds'] for r in run.read(OLD / 'budget_ledger.json')),
               total_new_simulator_batches=20, extra_candidates_per_input=0,
               budget_reason='Actual measured throughput exceeds earlier 1800s allowance; no formal feedback comparison started')
    run.save(OUT / 'versions.json', old, True)
    run.save(OUT / 'budget_ledger.json', [], True)
    run.validate()
    carried_first()
    print('TASK_MATCHED_CONTINUATION_PREPARED', flush=True)


def check():
    run.validate(); carried_first()
    version = run.read(OUT / 'versions.json')
    if version['batch_implementation_sha256'] != run.object_hash(assembled):
        raise RuntimeError('Assembled dispatcher changed')
    run.save(OUT / 'runtime_witness.json', dict(status='PASS', versions_sha256=run.sha(OUT / 'versions.json'),
             previous_actual_runtime_witness_sha256=run.sha(OLD / 'runtime_witness.json'),
             previous_document_witness_sha256=run.sha(OLD / 'DOCUMENT_WITNESS.json'),
             new_gpu_batches=0, scope='Dispatch binding only; unchanged collector uses preceding actual three-batch witness'), True)
    print('TASK_MATCHED_CONTINUATION_CHECK_PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); g = p.add_mutually_exclusive_group(required=True)
    for mode in ('prepare', 'check', 'run'): g.add_argument('--' + mode, action='store_true')
    a = p.parse_args()
    if a.prepare: prepare()
    elif a.check:
        with (OUT / 'queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            check()
    else:
        run.validate()
        w = run.read(OUT / 'DOCUMENT_WITNESS.json')
        if w['status'] != 'PASS' or w['runtime_witness_sha256'] != run.sha(OUT / 'runtime_witness.json'):
            raise RuntimeError('Independent dispatcher witness required')
        run.prior.prepare = run.validate
        run.prior.batch = batch
        run.prior.main(False)
        print('TASK_MATCHED_CONTINUATION_COMPLETE', flush=True)

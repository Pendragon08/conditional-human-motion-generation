"""Separate, bounded83703→101500 resume or disposable restoration witness."""
import argparse
import json
import os
import pathlib
import signal
import time
import traceback
from tmr100_capacity_resume_protocol import ROOT, PARENT, RUNS, owned, read, sha, origin, accepted_witnesses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--role', choices=list(RUNS), required=True)
    a = parser.parse_args()
    sp = owned(a.request)
    spec = read(sp)
    role, run = a.role, RUNS[a.role]
    assert sha(__file__) == spec['executor_sha256'] and os.getuid() == 1013
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1' and set(os.sched_getaffinity(0)) == {0, 1, 2, 3}
    assert spec['formal_training_authorized'] if role == 'train' else spec['witnesses_authorized']
    origin(spec)
    if role == 'train':
        dp = owned(spec['document_summary']['path'])
        assert sha(dp) == spec['document_summary']['sha256']
        assert accepted_witnesses(read(dp)) == spec['witness_bindings']
    out = owned(PARENT) / run
    out.mkdir(mode=0o700, exist_ok=False)
    started, session, metrics = time.monotonic(), None, None
    rec = dict(status='PREPARING_TMR100_CAPACITY_RESUME', run=run, role=role,
        source_sha256=sha(__file__), request_sha256=sha(sp), runtime_binding_sha256=spec['runtime_binding_sha256'],
        output_writers_closed=False, formal_optimizer_updates=83703, completed_epochs=82,
        actual_new_formal_optimizer_updates=0, attempted_new_optimizer_updates=0, skipped_updates=0,
        target_epochs=100, expected_optimizer_updates=101500, original_failed_attempts_retained=83704,
        generator_or_teacher_training_updates=0, original_validation_sources_used_for_gradient=0,
        checkpoints=[], semantic_metrics_computed=False)

    def save(**values):
        rec.update(values)
        rec['elapsed_seconds'] = time.monotonic() - started
        temporary = out / 'receipt.partial'
        with temporary.open('w') as f:
            json.dump(rec, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
        temporary.replace(out / 'receipt.json')

    def interrupted(number, frame):
        raise RuntimeError('Owned TMR100 capacity resume interrupted with signal {}'.format(number))

    for number in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(number, interrupted)
    save()
    try:
        from tmr100_capacity_resume_runtime import Session
        session = Session(spec, sp, save)
        if role != 'train':
            result = session.witness()
            save(status='TMR100_CAPACITY_RESUME_REAL_GPU_WITNESS_VERIFIED', output_writers_closed=True, **result)
            return
        session.home_guard()
        metrics = (out / 'metrics.jsonl').open('x')
        save(status='RUNNING_TMR100_CAPACITY_RESUMED_TRAINING',
            restored_from_safe_optimizer_updates=83703, initial_completed_epochs=82,
            original_class_AST_audit=session.ast_audit, selected_source_checkpoint=session.source)
        while session.completed < 100:
            if session.pending:
                validation = session.finish_validation()
                rec['last_internal_validation_loss'] = validation
                rec['internal_validation_optimizer_updates'] = 0
                metrics.write(json.dumps(dict(kind='internal_validation', completed_epochs=session.completed,
                    updates=session.updates, losses=validation, actual_optimizer_updates=0, training_rng_restored=True)) + '\n')
                metrics.flush()
                save(completed_epochs=session.completed, pending_validation=False)
                continue
            rec['attempted_new_optimizer_updates'] += 1
            values, batch_sha = session.update()
            rec.update(formal_optimizer_updates=session.updates, actual_new_formal_optimizer_updates=session.updates - 83703,
                       completed_epochs=session.completed, pending_validation=session.pending)
            if session.updates % 100 == 0 or session.pending:
                session.home_guard()
                metrics.write(json.dumps(dict(kind='train', updates=session.updates, completed_epochs=session.completed,
                    batch_sha256=batch_sha, losses=values), allow_nan=False) + '\n')
                metrics.flush()
                save(last_train_loss=values, sampler_epoch=session.sampler.epoch, sampler_cursor=session.sampler.cursor)
        assert session.updates == 101500 and session.pending is False
        assert rec['attempted_new_optimizer_updates'] == rec['actual_new_formal_optimizer_updates'] == 17797
        session.unchanged()
        item = session.save_checkpoint()
        rec['checkpoints'].append(item)
        metrics.flush(); os.fsync(metrics.fileno()); metrics.close(); metrics = None
        save(status='TMR100_CAPACITY_RESUMED_100_EPOCHS_FINISHED', output_writers_closed=True,
            selected_checkpoint=item, completed_epochs=100, formal_optimizer_updates=101500,
            final_model_sha256=session.module_sha(session.model), source_inputs_and_checkpoint_unchanged=True,
            exact_final_model_adamw_all_rng_sampler_nextbatch_disk_reload=True, original500_success_claimed=False)
    except BaseException as error:
        save(status='FAILED_TMR100_CAPACITY_RESUME_ALL_STATES_RETAINED', error=repr(error), traceback=traceback.format_exc())
        if role == 'train' and session is not None and session.last_good is not None:
            try:
                item = session.save_checkpoint(emergency=True)
                rec['emergency_checkpoint'] = item
            except BaseException as emergency_error:
                rec['emergency_error'] = repr(emergency_error)
        if metrics is not None:
            metrics.flush(); metrics.close()
        save()
        raise


if __name__ == '__main__':
    main()

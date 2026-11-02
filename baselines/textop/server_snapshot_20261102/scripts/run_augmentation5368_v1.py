"""Bounded TRAIN-only generated-motion job; invoke through its supervisor."""
import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import time
import traceback

from filtered15000_vae_completion import ROOT, owned, read, sha


def atomic_json(path, obj):
    temp = path.with_name(path.name + '.partial')
    with temp.open('xb') as stream:
        os.chmod(temp, 0o600)
        stream.write((json.dumps(obj, indent=2, allow_nan=False) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def require_supervisor(request):
    pid = os.getppid()
    for _ in range(8):
        if pid <= 1:
            break
        proc = pathlib.Path('/proc') / str(pid)
        argv = (proc / 'cmdline').read_bytes().decode().split('\0')
        if (proc.stat().st_uid == 1013 and str(request) in argv and
                str(ROOT / 'scripts/supervise_augmentation5368_v1.py') in argv):
            break
        pid = int((proc / 'stat').read_text().rsplit(')', 1)[1].split()[1])
    else:
        raise AssertionError('supervisor ancestor not found')
    assert pid > 1 and str(request) in argv
    with (ROOT / 'receipts/full_library_capacity.lock').open('r+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
    raise AssertionError('common model lock is not held')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    request = owned(args.request)
    s = read(request)
    assert os.getuid() == 1013 and os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
    assert set(os.sched_getaffinity(0)) == {0, 1, 2, 3}
    assert s['schema'] == 'textop.augmentation5368_job.v1' and s['authorized'] is True
    assert s['role'] in ('author', 'document', 'production')
    assert sha(__file__) == s['executor_sha256']
    require_supervisor(request)
    for name, digest in s['bound_sources'].items():
        assert sha(owned(ROOT / name)) == digest
    from augmentation5368_verified_inputs_v1 import accepted_inputs
    actual = accepted_inputs()
    clips = actual['plan']['clips']
    assert len(clips) == 5368
    if s['role'] == 'production':
        # This separate acceptance module is deliberately required; the
        # production request must bind two actual closed numerical witnesses.
        from augmentation5368_witness_acceptance_v1 import accepted_witnesses
        accepted_witnesses(s['witness_binding'])
    else:
        assert s['witness_binding'] is None
        clips = clips[:1]
    output_limit = sum(c['target_pose_frames'] * 152 + 4096 for c in clips) + 128 * 1024**2
    assert output_limit == s['output_upper_bytes']
    assert shutil.disk_usage(ROOT).free >= s['reserve_bytes'] + s['log_allowance_bytes'] + output_limit
    parent = owned(ROOT / 'step34_20261029')
    out = parent / s['run']
    out.mkdir(mode=0o700, exist_ok=False)
    start = time.monotonic()
    receipt = dict(status='STARTING_TRAIN_AUGMENTATION', run=s['run'], role=s['role'],
        source_sha256=sha(__file__), request_sha256=sha(request),
        scope_receipt_sha256=actual['scope_receipt_sha256'], scope_data_sha256=actual['scope_data_sha256'],
        selected_generator_receipt_sha256=actual['verified']['receipt_sha256'],
        planned_clips=len(clips), completed_clips=0, completed_intervals=0,
        training_optimizer_updates=0, output_writers_closed=False, real_generated_reference=False)
    def save():
        receipt['elapsed_seconds'] = time.monotonic() - start
        atomic_json(out / 'receipt.json', receipt)
    save()
    try:
        from augmentation5368_runtime import Runtime
        runtime = Runtime(actual['verified'], actual['train_index'])
        receipt.update(status='RUNNING_TRAIN_AUGMENTATION', generator_model_sha256=runtime.model_hash,
                       vae_model_sha256=runtime.vae_hash, actual_inference_model_loads=2)
        save()
        journal_path = out / 'clips.private.jsonl'
        with journal_path.open('x', encoding='utf-8') as journal:
            os.chmod(journal_path, 0o600)
            for clip in clips:
                # Reserve the complete declared output, not a logical-file-size
                # subtraction; free-space samples are never reservations.
                assert shutil.disk_usage(ROOT).free >= s['reserve_bytes'] + s['log_allowance_bytes'] + output_limit
                index = clip['clip_index']
                journal.write(json.dumps(dict(index=index, status='STARTED')) + '\n')
                journal.flush(); os.fsync(journal.fileno())
                arrays, proof = runtime.generate(clip)
                filename = 'clip_' + str(index).zfill(5) + '.private.npz'
                path = out / filename
                tmp = out / (filename + '.partial')
                with tmp.open('xb') as stream:
                    os.chmod(tmp, 0o600)
                    runtime.np.savez(stream, **arrays)
                    stream.flush(); os.fsync(stream.fileno())
                limit = clip['target_pose_frames'] * 152 + 4096
                assert 0 < tmp.stat().st_size <= limit
                with runtime.np.load(tmp, allow_pickle=False) as loaded:
                    assert set(loaded.files) == set(arrays)
                    for name in arrays:
                        assert runtime.np.array_equal(loaded[name], arrays[name])
                tmp.replace(path)
                directory_fd = os.open(str(out), os.O_RDONLY)
                try: os.fsync(directory_fd)
                finally: os.close(directory_fd)
                item = dict(index=index, status='COMPLETED', file=str(path), bytes=path.stat().st_size,
                    sha256=sha(path), array_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()},
                    proof=proof)
                journal.write(json.dumps(item, allow_nan=False) + '\n'); journal.flush(); os.fsync(journal.fileno())
                receipt['completed_clips'] += 1
                receipt['completed_intervals'] += proof['control_intervals']
                save()
        runtime.verify_unchanged()
        receipt.update(status='TRAIN_AUGMENTATION_FINISHED', output_writers_closed=True,
                       real_generated_reference=True, journal_sha256=sha(journal_path))
        save()
    except BaseException as error:
        receipt.update(status='TRAIN_AUGMENTATION_FAILED_PARTIAL_RETAINED', error=repr(error),
                       traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()

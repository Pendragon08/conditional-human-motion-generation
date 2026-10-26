"""Bound a newly spawned evaluator and persist samples, including failure exits.

This module never discovers or stops an existing job. Signals target only the
still-owned process group created here. It has no SDK/Torch imports.
"""
import json
import os
import pathlib
import signal
import subprocess
import time
import traceback


class SupervisorSignal(Exception):
    pass


def run_supervised(command, log_path, samples_path, proof_path, sampler,
                   timeout_seconds, sample_interval=15.0, grace_seconds=10.0,
                   affinity=(0, 1, 2, 3)):
    assert os.name == 'posix'
    assert timeout_seconds > 0 and sample_interval > 0 and grace_seconds > 0
    paths = [pathlib.Path(p) for p in (log_path, samples_path, proof_path)]
    assert len(set(paths)) == 3 and all(not p.exists() for p in paths)
    assert all(p.parent.resolve() == p.parent and p.parent.stat().st_uid == os.getuid()
               for p in paths), 'Outputs must be new files in owned persistent folders'
    started = time.monotonic()
    child = None
    proof = dict(command=list(command), supervisor_pid=os.getpid(),
                 child_pid=None, child_exit_code=None, child_reaped=False,
                 timed_out=False, supervisor_error=None, shutdown_actions=[],
                 samples_written=0, sample_errors=[])
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}

    def interrupted(signum, frame):
        raise SupervisorSignal('Supervisor received signal {}'.format(signum))

    def stop_owned_child():
        if child is None or child.poll() is not None:
            return
        assert os.getpgid(child.pid) == child.pid, 'Unexpected child group; refusing broad signal'
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if child.poll() is not None:
                break
            try:
                os.killpg(child.pid, sig)
                proof['shutdown_actions'].append({'owned_group': child.pid, 'signal': sig.name})
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=grace_seconds)
                break
            except subprocess.TimeoutExpired:
                if sig == signal.SIGKILL:
                    raise

    with paths[0].open('xb') as log, paths[1].open('x', encoding='utf-8') as samples:
        def sample(phase):
            try:
                row = sampler(child.pid if child is not None else None)
                row = dict(row, phase=phase, elapsed_seconds=time.monotonic() - started)
                samples.write(json.dumps(row, allow_nan=False) + '\n')
                samples.flush()
                proof['samples_written'] += 1
            except Exception as error:
                proof['sample_errors'].append({'phase': phase, 'error': repr(error)})
                raise

        try:
            for sig in previous:
                signal.signal(sig, interrupted)
            sample('before_launch')
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True,
                                     preexec_fn=lambda: os.sched_setaffinity(0, set(affinity)))
            proof['child_pid'] = child.pid
            sample('after_launch')
            next_sample = time.monotonic() + sample_interval
            while child.poll() is None:
                now = time.monotonic()
                if now - started >= timeout_seconds:
                    proof['timed_out'] = True
                    break
                if now >= next_sample:
                    sample('running')
                    next_sample = time.monotonic() + sample_interval
                try:
                    child.wait(timeout=max(0.01, min(0.5, next_sample - time.monotonic(),
                                                    timeout_seconds - (time.monotonic() - started))))
                except subprocess.TimeoutExpired:
                    pass
        except BaseException as error:
            proof['supervisor_error'] = repr(error)
            proof['supervisor_traceback'] = traceback.format_exc()
        finally:
            # Finish handling the first signal without abandoning the owned child.
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
            try:
                stop_owned_child()
                if child is not None:
                    proof['child_exit_code'] = child.wait(timeout=grace_seconds)
                    proof['child_reaped'] = True
                try:
                    sample('after_exit')
                except Exception as error:
                    if proof['supervisor_error'] is None:
                        proof['supervisor_error'] = repr(error)
            except BaseException as error:
                proof['cleanup_error'] = repr(error)
            finally:
                proof['elapsed_seconds'] = time.monotonic() - started
                proof['supervision_passed'] = (proof['child_reaped'] and
                    proof['child_exit_code'] == 0 and not proof['timed_out'] and
                    proof['supervisor_error'] is None and not proof.get('cleanup_error'))
                with paths[2].open('x', encoding='utf-8') as stream:
                    json.dump(proof, stream, indent=2, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
    return proof

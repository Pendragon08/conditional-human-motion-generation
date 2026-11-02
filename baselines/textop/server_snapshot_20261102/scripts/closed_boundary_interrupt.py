"""Interrupt one SHA-bound trainer only while its storage-guard child is held.

No code injection, source edits, broad signals, or forced trainer termination.
The bound launcher calls findmnt with at_closed_boundary=True every100 steps.
Holding that exact direct child prevents return from that verified boundary.
SIGINT then enters the launcher's existing full-state emergency-save handler.
This is an explicit user pause, not a numerical or storage failure.
"""
import ctypes, errno, json, os, pathlib, re, signal, time
from quick_pidfd import BoundProcess as OriginalBoundProcess, start_ticks

# Cache the same verified x86_64 syscall metadata used by quick_pidfd. Reading
# the header and opening libc for every signal can outlive a findmnt process.
assert os.uname().machine == 'x86_64'
_header = pathlib.Path('/usr/include/x86_64-linux-gnu/asm/unistd_64.h').read_text()
_numbers = {n: int(re.search(r'^#define __NR_' + n + r'\s+(\d+)$', _header, re.MULTILINE).group(1))
            for n in ('pidfd_open', 'pidfd_send_signal')}
_libc = ctypes.CDLL(None, use_errno=True); _libc.syscall.restype = ctypes.c_long

def _call(name, *args):
    result = _libc.syscall(ctypes.c_long(_numbers[name]), *args)
    if result < 0:
        code = ctypes.get_errno()
        if code == errno.ESRCH: raise ProcessLookupError(code, os.strerror(code))
        raise OSError(code, os.strerror(code))
    return result

class BoundProcess(OriginalBoundProcess):
    def __init__(self, pid, expected_start_ticks):
        self.pid = pid; self.fd = _call('pidfd_open', ctypes.c_int(pid), ctypes.c_uint(0))
        try: assert start_ticks(pid) == expected_start_ticks, 'PID identity changed'
        except BaseException: os.close(self.fd); raise

    def send(self, sig):
        if not self.alive(): raise ProcessLookupError('Original process exited')
        _call('pidfd_send_signal', ctypes.c_int(self.fd), ctypes.c_int(sig), ctypes.c_void_p(0), ctypes.c_uint(0))


def identity(pid):
    p = pathlib.Path('/proc') / str(pid)
    stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': int(stat[19]), 'ppid': int(stat[1]),
            'session': int(stat[3]), 'state': stat[0], 'uid': p.stat().st_uid,
            'argv': (p / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
            'exe': str((p / 'exe').resolve())}


def require_identity(actual, expected):
    for key in ('pid', 'start_ticks', 'ppid', 'session', 'uid', 'argv', 'exe'):
        assert actual[key] == expected[key], 'Bound identity changed: ' + key


def interrupt_at_guard(expected, child_argv, child_exe, journal,
                       watch_seconds=300, exit_seconds=600):
    assert os.getuid() == expected['uid'] == 1013
    assert child_argv[0] == 'findmnt' and child_argv[-3:] == ['-n', '-o', 'FSTYPE']
    assert watch_seconds > 0 and exit_seconds > 0
    path = pathlib.Path(journal)
    assert path.parent.resolve() == path.parent and path.parent.stat().st_uid == os.getuid()
    stream = path.open('x', encoding='utf-8')  # Existing/pending sequence is never replayed.
    worker = BoundProcess(expected['pid'], expected['start_ticks'])
    child = None
    stopped_worker = False
    stopped_child = False
    interrupt_sent = False
    started = time.monotonic()
    previous = {}

    def record(event, sync=True, **fields):
        stream.write(json.dumps(dict(event=event, monotonic=time.monotonic(), **fields)) + '\n')
        stream.flush()
        if sync: os.fsync(stream.fileno())

    def send(bound, sig, role):
        # The trainer's interruption is always fsync-before-signal. For the
        # very short-lived utility only, flush its STOP intent then fsync both
        # records after STOP; otherwise the fsync itself misses the utility.
        record('signal_intent', sync=not(role=='direct_storage_guard_child' and sig==signal.SIGSTOP), role=role, pid=bound.pid, signal=sig.name)
        bound.send(sig)
        record('signal_sent', role=role, pid=bound.pid, signal=sig.name)

    def wait_stopped(bound):
        limit = time.monotonic() + 1
        while time.monotonic() < limit:
            if not bound.alive(): return False
            try:
                if identity(bound.pid)['state'] in ('T', 't'): return True
            except (FileNotFoundError, ProcessLookupError): return False
            time.sleep(.001)
        raise TimeoutError('Bound process did not enter stopped state')

    def interrupted(signum, frame):
        raise RuntimeError('Controller interrupted: ' + str(signum))

    try:
        require_identity(identity(worker.pid), expected)
        assert identity(worker.pid)['state'] not in ('T', 't'), 'Do not resume an already-paused process'
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.getsignal(sig); signal.signal(sig, interrupted)
        record('armed', worker=expected, exact_child_argv=child_argv)
        while time.monotonic() - started < watch_seconds:
            assert worker.alive(), 'Trainer already exited; no signal sent'
            kids = pathlib.Path('/proc/{0}/task/{0}/children'.format(worker.pid)).read_text().split()
            for raw in kids:
                pid = int(raw)
                try:
                    found = identity(pid)
                    if found['ppid'] != worker.pid or found['uid'] != expected['uid']:
                        continue
                    if found['argv'] != child_argv or found['exe'] != child_exe or found['state'] in ('T', 't'):
                        continue
                    child = BoundProcess(pid, found['start_ticks'])
                    send(child, signal.SIGSTOP, 'direct_storage_guard_child'); stopped_child = True
                    if not wait_stopped(child):
                        child.close(); child = None; stopped_child = False; continue
                    require_identity(identity(pid), found)
                    require_identity(identity(worker.pid), expected)
                    record('storage_guard_held', child=found, worker=identity(worker.pid))
                    send(worker, signal.SIGSTOP, 'trainer'); stopped_worker = True
                    assert wait_stopped(worker)
                    require_identity(identity(worker.pid), expected)
                    assert child.alive() and identity(child.pid)['state'] in ('T', 't')
                    record('verified_closed_boundary_interruption', no_source_or_math_change=True)
                    send(worker, signal.SIGINT, 'trainer'); interrupt_sent = True
                    send(worker, signal.SIGCONT, 'trainer'); stopped_worker = False
                    # Allow the child to finish if Python's subprocess cleanup has not killed it.
                    if child.alive():
                        try: send(child, signal.SIGCONT, 'direct_storage_guard_child')
                        except ProcessLookupError: pass
                    stopped_child = False
                    limit = time.monotonic() + exit_seconds
                    while worker.alive() and time.monotonic() < limit: time.sleep(.2)
                    record('trainer_exit_observed', exited=not worker.alive(), interrupt_sent=True)
                    assert not worker.alive(), 'Checkpoint/exit pending; retain journal and never repeat signals'
                    return {'status': 'BOUNDARY_SIGINT_SENT_TRAINER_EXITED', 'training_signal_sequences': 1,
                            'source_edits': 0, 'forced_trainer_kills': 0,
                            'checkpoint_acceptance': 'Requires independent actual file/receipt/exit verification'}
                except (FileNotFoundError, ProcessLookupError):
                    if interrupt_sent: raise
                    if stopped_worker: raise
                    if child is not None:
                        if stopped_child and child.alive():
                            send(child, signal.SIGCONT, 'direct_storage_guard_child_cleanup')
                        child.close(); child = None; stopped_child = False
            time.sleep(.001)
        raise TimeoutError('No exact guard child captured; trainer left running, no SIGINT')
    finally:
        # Never leave a process stopped solely by this controller on a handled error.
        for bound, stopped, role in ((worker, stopped_worker, 'trainer_cleanup'),
                                      (child, stopped_child, 'child_cleanup')):
            if bound is not None and stopped and bound.alive():
                try: send(bound, signal.SIGCONT, role)
                except ProcessLookupError: pass
        for sig, old in previous.items(): signal.signal(sig, old)
        if child is not None: child.close()
        worker.close(); stream.close()

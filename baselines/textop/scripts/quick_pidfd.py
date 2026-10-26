"""Bind signals to one original process, refusing PID reuse; Python3.8-compatible."""
import ctypes
import errno
import os
import pathlib
import re
import select
import signal


def start_ticks(pid):
    text = pathlib.Path('/proc/{}/stat'.format(pid)).read_text()
    return int(text.rsplit(')', 1)[1].split()[19])


def _syscall(name, *args):
    assert os.uname().machine == 'x86_64', 'No unverified syscall-number fallback'
    header = pathlib.Path('/usr/include/x86_64-linux-gnu/asm/unistd_64.h').read_text()
    number = int(re.search(r'^#define __NR_' + name + r'\s+(\d+)$', header, re.MULTILINE).group(1))
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *args)
    if result < 0:
        code = ctypes.get_errno()
        if code == errno.ESRCH:
            raise ProcessLookupError(code, os.strerror(code))
        raise OSError(code, os.strerror(code))
    return result


class BoundProcess:
    def __init__(self, pid, expected_start_ticks):
        self.pid = pid
        self.fd = os.pidfd_open(pid, 0) if hasattr(os, 'pidfd_open') else _syscall(
            'pidfd_open', ctypes.c_int(pid), ctypes.c_uint(0))
        try:
            assert start_ticks(pid) == expected_start_ticks, 'Process identity changed; no signal allowed'
        except BaseException:
            os.close(self.fd)
            raise

    def alive(self):
        return not select.select([self.fd], [], [], 0)[0]

    def send(self, sig):
        if not self.alive():
            raise ProcessLookupError('Original bound process already exited')
        if hasattr(signal, 'pidfd_send_signal'):
            signal.pidfd_send_signal(self.fd, sig, None, 0)
        else:
            _syscall('pidfd_send_signal', ctypes.c_int(self.fd), ctypes.c_int(sig),
                     ctypes.c_void_p(0), ctypes.c_uint(0))

    def close(self):
        os.close(self.fd)

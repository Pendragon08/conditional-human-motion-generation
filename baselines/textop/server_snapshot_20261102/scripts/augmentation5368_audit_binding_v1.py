"""Draft pure checks for a terminal audit's producer and lock binding.

These functions do not authorize or launch a job, read arrays, or acquire a
lock. A caller must construct the observations from trusted, stable files and
test the Linux process and lock behavior on the actual host before use.
"""
import re

_SHA = re.compile(r'^[0-9a-f]{64}$')
_FIELDS = frozenset(('request', 'receipt', 'exit', 'process', 'outer', 'resources'))


def validate_closure_binding(expected, observed):
    if not isinstance(expected, dict) or not isinstance(observed, dict):
        raise ValueError('a complete immutable closure binding is required')
    if set(expected) != _FIELDS or set(observed) != _FIELDS:
        raise ValueError('closure file set is incomplete or unexpected')
    for key in _FIELDS:
        if not isinstance(expected[key], str) or not _SHA.fullmatch(expected[key]):
            raise ValueError('invalid expected closure hash')
        if observed[key] != expected[key]:
            raise ValueError('producer closure hash changed: ' + key)
    return True


def lock_owned_by_supervisor(lines, supervisor_pid, device_major, device_minor, inode):
    """Check /proc/locks text against the same-namespace supervisor and inode.

    Only an exclusive whole-file FLOCK owned by this PID qualifies. Lock wait
    rows, POSIX locks, another owner, a different device/inode, and partial or
    multiple matching records fail. The caller must first establish that PID
    is the expected live ancestor and stat the intended owned lock file.
    """
    if any(type(x) is not int or x < 0 for x in (supervisor_pid, device_major, device_minor, inode)):
        raise ValueError('invalid process or file identity')
    if supervisor_pid <= 1 or inode == 0:
        raise ValueError('invalid supervisor or inode')
    matches = 0
    for line in lines:
        fields = line.split()
        if len(fields) != 8 or not fields[0].endswith(':'):
            continue
        if fields[1:4] != ['FLOCK', 'ADVISORY', 'WRITE']:
            continue
        try:
            pid = int(fields[4])
            major, minor, found_inode = fields[5].split(':')
            ident = (int(major, 16), int(minor, 16), int(found_inode, 10))
        except ValueError:
            continue
        if ident != (device_major, device_minor, inode):
            continue
        if pid != supervisor_pid or fields[6:] != ['0', 'EOF']:
            return False
        matches += 1
    return matches == 1

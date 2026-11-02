"""Isolated float32 NPY corruption checks; no real outputs, models or authority claims."""
import hashlib
import json
import pathlib
import struct
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from ldm100k_generation_artifacts import check_npy


def make(values, shape=(2, 2), descr='<f4', order=False, extra=None):
    header = dict(descr=descr, fortran_order=order, shape=shape)
    if extra:
        header.update(extra)
    h = repr(header).encode()
    h += b' ' * ((-(10 + len(h) + 1)) % 64) + b'\n'
    data = struct.pack('<' + 'f' * len(values), *values)
    return b'\x93NUMPY\x01\x00' + len(h).to_bytes(2, 'little') + h + data, hashlib.sha256(data).hexdigest()


raw, digest = make([0, .25, -1, 2])
assert check_npy(raw, [2, 2], digest) == [0, .25, -1, 2]
count = 0


def reject(data, shape, checksum):
    global count
    try:
        check_npy(data, shape, checksum)
    except (AssertionError, ValueError, SyntaxError, KeyError, IndexError, TypeError):
        count += 1
        return
    raise AssertionError('Corrupt isolated NPY accepted')


reject(b'WRONG!' + raw[6:], [2, 2], digest)
reject(raw[:6] + b'\x04\x00' + raw[8:], [2, 2], digest)
reject(raw[:-1], [2, 2], digest)
reject(raw + b'x', [2, 2], digest)
reject(raw, [4, 1], digest)
reject(raw, [2, True], digest)
reject(raw, [2, 2], '0' * 64)
for kwargs in [dict(descr='>f4'), dict(descr='<f8'), dict(descr='|O'),
               dict(order=True), dict(extra={'unexpected': True})]:
    altered, checksum = make([0, .25, -1, 2], **kwargs)
    reject(altered, [2, 2], checksum)
for value in [float('inf'), float('nan')]:
    altered, checksum = make([value, .25, -1, 2])
    reject(altered, [2, 2], checksum)
print(json.dumps(dict(status='LDM100K_ARTIFACT_NEW_ISOLATED_NPY_FIXTURES_PASSED',
    negative_cases=count, production_array_payloads_read=0, actual_model_loads=0,
    actual_gpu_contexts=0, real_witnesses_accepted=0, dynamic_posix_ownership_tested=False)))

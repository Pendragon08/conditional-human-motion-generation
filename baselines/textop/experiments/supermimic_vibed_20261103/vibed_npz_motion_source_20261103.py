"""Load one validated 50 Hz TextOp NPZ without creating an intermediate CSV."""
import hashlib
import io
import os
import stat
import zipfile

import numpy as np

from vibed_motion_adapter_50hz_20261103 import motion_to_csv_matrix


def load_motion(record):
    required = {'path', 'sha256', 'bytes', 'frames', 'joint_names', 'body_names'}
    if set(record) != required or not isinstance(record['path'], str):
        raise ValueError('invalid motion record')
    if not isinstance(record['bytes'], int) or not 0 < record['bytes'] <= 16 * 1024 * 1024:
        raise ValueError('invalid file size')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(record['path'], flags)
    with os.fdopen(fd, 'rb') as f:
        before = os.fstat(f.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != record['bytes']:
            raise ValueError('source file changed')
        data = f.read(record['bytes'] + 1)
        after = os.fstat(f.fileno())
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, x) != getattr(after, x) for x in fields):
        raise ValueError('source changed while reading')
    if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError('source digest mismatch')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) > 32 or sum(x.file_size for x in members) > 64 * 1024 * 1024:
            raise ValueError('motion archive exceeds size limits')
        if len({x.filename for x in members}) != len(members):
            raise ValueError('duplicate archive entries')
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        arrays = {k: z[k] for k in ('fps', 'joint_pos', 'body_pos_w', 'body_quat_w')}
    result = motion_to_csv_matrix(arrays, record['joint_names'], record['body_names'])
    if result.shape != (record['frames'], 36):
        raise ValueError('frame count mismatch')
    return result

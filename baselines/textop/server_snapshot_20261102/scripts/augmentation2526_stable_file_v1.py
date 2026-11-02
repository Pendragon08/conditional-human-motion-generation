"""Stable, owned file hashing for a future mixed-policy audit.

This helper does not acquire the model lock, load a checkpoint, or authorize a
job. Its caller must establish the provenance and exclusive execution context.
"""
import hashlib
import os
import pathlib
import stat


def _identity(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_uid)


def stable_owned_sha256(path, uid):
    path = pathlib.Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_uid != uid:
        raise ValueError('file must be regular and owned by expected user')
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(str(path), flags)
    try:
        opened = os.fstat(fd)
        if _identity(before) != _identity(opened):
            raise ValueError('file identity changed before opening')
        digest = hashlib.sha256()
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        if _identity(opened) != _identity(os.fstat(fd)) or _identity(opened) != _identity(path.lstat()):
            raise ValueError('file changed during hashing')
        return {'path': str(path), 'bytes': opened.st_size, 'sha256': digest.hexdigest(),
                'device': opened.st_dev, 'inode': opened.st_ino,
                'mtime_ns': opened.st_mtime_ns, 'ctime_ns': opened.st_ctime_ns}
    finally:
        os.close(fd)

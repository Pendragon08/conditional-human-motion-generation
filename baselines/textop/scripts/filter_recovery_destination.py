"""Reject link/foreign/volatile output ancestry before an audit can write."""
import os
import pathlib
import subprocess


def guard_destination(root, destination, require_absent=True, owner_uid=1013):
    root, destination = pathlib.Path(root), pathlib.Path(destination)
    assert root.is_absolute() and root.resolve() == root and root.is_dir()
    assert root.stat().st_uid == owner_uid and os.access(root, os.W_OK)
    assert destination.is_absolute() and destination != root
    relative = destination.relative_to(root)
    assert relative.parts and '..' not in relative.parts
    assert destination.resolve() == destination, 'Output path contains a link or alias'
    filesystem = subprocess.check_output(['findmnt', '-T', str(root), '-n', '-o', 'FSTYPE'],
        universal_newlines=True, timeout=10).strip()
    assert filesystem in ('ext4', 'xfs', 'btrfs'), 'Persistent audit filesystem required'
    current = root
    existing = []
    for part in relative.parts:
        current = current / part
        if os.path.lexists(str(current)):
            assert not current.is_symlink() and current.resolve() == current
            st = current.stat()
            assert current.is_dir() and st.st_uid == owner_uid and st.st_dev == root.stat().st_dev
            assert os.access(current, os.W_OK)
            existing.append(str(current))
    if require_absent:
        assert not os.path.lexists(str(destination)), 'Preserve earlier output; never rerun into it'
    else:
        assert destination.is_dir()
    return dict(destination=str(destination), filesystem=filesystem, owner_uid=owner_uid,
                existing_owned_ancestors=existing, new_output_required=require_absent)

"""Exact-file cleanup of inactive caches and old transfer packages, after backup.

Standard library only. Never removes directories, data, checkpoints, records,
environments, live source, or files outside the one authorized server folder.
The client must independently verify the published archive before --execute.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat

ROOT = Path('/data/code/ouyanghao/oyh')
TEXTURES = [
    'phygile-unitree-20261022/code_v1/cache/xdg/ov/texturecache',
    'humanoid-combat-motion/cache/r3_xdg/ov/texturecache',
    'humanoid-combat-motion/cache/r3_xdg_gpu1/ov/texturecache',
    'humanoid-combat-motion/cache/r3_viewer_home/.cache/ov/texturecache',
    'conditional-human-motion-generation/cache/r3_punch_viewer_home/.cache/ov/texturecache',
    'conditional-human-motion-generation/cache/r3_i1_viewer_stream/xdg/ov/texturecache',
]
SHADERS = [
    'humanoid-combat-motion/cache/r3_xdg/ov/shaders/nv_shadercache',
    'humanoid-combat-motion/cache/r3_viewer_kit/cache/shadercache',
    'conditional-human-motion-generation/cache/r3_official_kit/cache/shadercache',
    'conditional-human-motion-generation/cache/r3_punch_viewer_kit/cache/shadercache',
]
PACKAGES = {
    'world-model-risk-planning/wheelhouse_sync.tar',
    'world-model-risk-planning/upstream_sync.tar.gz',
    'maniskill-hab-demo/code_maniskill.tar',
    'maniskill-hab-demo/code_mshab.tar',
}
ENV_KEYS = {'XDG_CACHE_HOME', 'XDG_DATA_HOME', 'XDG_CONFIG_HOME', 'HOME', 'TMPDIR',
    'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH', 'PIP_CACHE_DIR', 'PYTHONPYCACHEPREFIX'}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def below(path, parent):
    return path == parent or parent in path.parents


def permitted(path):
    require(path.is_absolute() and below(path, ROOT), 'Outside authorized root')
    relative = str(path.relative_to(ROOT))
    if relative in PACKAGES:
        return 'unused_transfer_package'
    for entry in TEXTURES:
        if path.parent == ROOT/entry and re.fullmatch(r'Wireframe[^/]*\.ovtex\.\d+', path.name):
            return 'inactive_builtin_texture_cache'
    for entry in SHADERS:
        if below(path, ROOT/entry) and path.suffix.lower() in {'.cache', '.bin', '.toc', '.idx'}:
            return 'inactive_shader_cache'
    raise ValueError('Not an explicitly approved cache/package file: '+str(path))


def identity(path, row=None):
    require(ROOT.resolve() == ROOT and os.getuid() == 1013, 'Unexpected user or root')
    permitted(path)
    require(path.resolve() == path, 'Symlink target rejected')
    for p in [path, *path.parents]:
        require(not p.is_symlink(), 'Symlink ancestor rejected')
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid() and s.st_nlink == 1,
            'Only singly-linked regular files belonging to this user')
    if row is not None:
        require((s.st_ino, s.st_mtime_ns, s.st_size) == (row['inode'], row['mtime_ns'], row['bytes']),
                'Candidate changed since backup')
    return s


def inactive(paths):
    names = set(map(str, paths))
    # A transfer package's parent may contain a live installation; only the
    # package itself matters there. Cache roots require no live cache-dir use.
    caches = {str(ROOT/x) for x in TEXTURES+SHADERS if any(below(p, ROOT/x) for p in paths)}
    helpers = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        args = []; comm = ''
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            args = [s for s in (proc/'cmdline').read_bytes().decode(errors='replace').split(chr(0)) if s]
            comm = (proc/'comm').read_text().strip()
            require(not any(name in arg for arg in args for name in names), 'Candidate appears in active process')
            require(not any(cache in arg for arg in args for cache in caches), 'Cache directory appears in active process')
            cwd = os.readlink(proc/'cwd')
            require(not any(cwd == c or cwd.startswith(c+'/') for c in caches), 'Active cache working directory')
            for part in (proc/'environ').read_bytes().decode(errors='replace').split(chr(0)):
                key, _, value = part.partition('=')
                if key in ENV_KEYS and value.startswith('/'):
                    value = value.rstrip('/')
                    require(not any(c == value or c.startswith(value+'/') or value.startswith(c+'/') for c in caches),
                            'Candidate cache directory belongs to a live process')
            for fd in (proc/'fd').iterdir():
                try:
                    value = os.readlink(fd)
                except FileNotFoundError:
                    continue
                require(value not in names and not any(value == c or value.startswith(c+'/') for c in caches),
                        'Candidate file/cache open by a live process')
            for line in (proc/'maps').read_text().splitlines():
                if '/' in line:
                    value = '/'+line.split('/', 1)[1]
                    require(value not in names and not any(value.startswith(c+'/') for c in caches),
                            'Candidate mapped by a live process')
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            known = {('(sd-pam)', ('(sd-pam)',)), ('ssh-agent', ('/bin/ssh-agent', '-s')),
                ('gpg-agent', ('/usr/bin/gpg-agent', '--supervised')),
                ('gnome-keyring-d', ('/usr/bin/gnome-keyring-daemon', '--start', '--foreground', '--components=secrets')),
                ('sshd', ('sshd: ouyanghao@notty',))}
            require((comm, tuple(args)) in known, 'Unknown unreadable same-user process; refusing cleanup: '+comm)
            helpers.append({'pid': int(proc.name), 'comm': comm, 'reason': 'OS protected session/key helper'})
    return helpers


def save(path, obj):
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(obj, indent=2)+'\n')
    os.replace(tmp, path)


def main(a):
    require(not a.receipt.exists(), 'Receipt already exists')
    plan = json.loads(a.plan.read_text())
    rows = plan['files']; paths = [Path(r['path']) for r in rows]
    require(rows and len(paths) == len(set(paths)), 'Empty or repeated file list')
    backup = plan['backup_verification']
    prefix = 'https://github.com/Pendragon08/conditional-human-motion-generation/releases/download/'
    if backup.get('mode') == 'direct_file_assets':
        require(backup['all_local_copies_verified'] is True, 'Local direct-file backup unverified')
        assets = {x['source_path']: x for x in backup['assets']}
        require(set(assets) == set(map(str, paths)), 'Direct backup set differs from cleanup set')
        for row in rows:
            asset = assets[row['path']]
            require(asset['url'].startswith(prefix), 'Wrong backup repository')
            require(asset['bytes'] == row['bytes'] and asset['digest'] == 'sha256:'+row['sha256'],
                    'Direct GitHub backup bytes or digest mismatch')
    else:
        require(backup['archive_contents_recovered_and_verified'] is True, 'No completed local recovery verification')
        require(backup['asset']['url'].startswith(prefix), 'Wrong backup repository')
        require(backup['asset']['digest'] == 'sha256:'+backup['local_archive_sha256'], 'GitHub archive digest mismatch')
        require(backup['verified_member_paths'] == [r['archive_member'] for r in rows], 'Recovered set differs from cleanup set')
    helpers = inactive(paths)
    total = 0
    for path, row in zip(paths, rows):
        s = identity(path, row)
        require(digest(path) == row['sha256'], 'File differs from recovered backup')
        total += s.st_blocks*512
    helpers += inactive(paths)
    receipt = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'mode': 'EXECUTE' if a.execute else 'DRY_RUN', 'plan_sha256': digest(a.plan),
        'authorized_root': str(ROOT), 'verified_files': len(rows), 'allocated_bytes': total,
        'disk_before': shutil.disk_usage('/data')._asdict(), 'removed': [],
        'known_os_helpers': list({x['pid']: x for x in helpers}.values()),
        'backup_verification': backup, 'dataset_checkpoint_record_deletion_allowed': False}
    a.receipt.parent.mkdir(parents=True, exist_ok=True); save(a.receipt, receipt)
    if a.execute:
        for path, row in zip(paths, rows):
            identity(path, row)
            path.unlink()
            receipt['removed'].append({'path': str(path), 'bytes': row['bytes'], 'sha256': row['sha256'],
                                       'kind': permitted(path)})
            save(a.receipt, receipt)
    receipt.update(status='BACKED_UP_UNUSED_FILES_REMOVED' if a.execute else 'DRY_RUN_PASSED',
        disk_after=shutil.disk_usage('/data')._asdict())
    save(a.receipt, receipt)
    print(json.dumps({'status': receipt['status'], 'files': len(receipt['removed']),
        'allocated_bytes': total, 'free_after': receipt['disk_after']['free']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    main(p.parse_args())

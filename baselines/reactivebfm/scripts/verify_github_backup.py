"""Verify pinned GitHub Git objects and server-computed release asset digests.

Requires authenticated gh and local git checkout. Writes a NEW receipt outside
the verified Git snapshot, avoiding a self-referential manifest.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

p = argparse.ArgumentParser()
p.add_argument('--checkout', type=Path, required=True)
p.add_argument('--commit', required=True)
p.add_argument('--repo', default='Pendragon08/conditional-human-motion-generation')
p.add_argument('--tag', default='reactivebfm-pilot-20261021')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--method', choices=['git-tree', 'archive'], default='git-tree')
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
prefix = 'baselines/reactivebfm/'
def command(args):
    return subprocess.check_output(args, cwd=a.checkout)
commit = command(['git', 'rev-parse', a.commit]).decode().strip()
paths = command(['git', 'ls-tree', '-rz', '--name-only', commit, '--', prefix]).decode().rstrip('\0').split('\0')
if a.method == 'git-tree':
    expected = {}
    rows = command(['git', 'ls-tree', '-rz', commit, '--', prefix]).decode().rstrip('\0').split('\0')
    for row in rows:
        info, path = row.split('\t', 1)
        mode, kind, oid = info.split()
        if kind == 'blob':
            expected[path] = oid
    remote = json.loads(command(['gh', 'api', f'repos/{a.repo}/git/trees/{commit}?recursive=1']))
    if remote.get('truncated'):
        raise RuntimeError('GitHub tree truncated; cannot validate the full file set')
    seen = {r['path']: r['sha'] for r in remote['tree'] if r['type'] == 'blob' and r['path'].startswith(prefix)}
else:
    expected = {s: hashlib.sha256(command(['git', 'show', commit + ':' + s])).hexdigest() for s in paths}
    seen = {}
    with tempfile.TemporaryFile() as archive:
        subprocess.run(['gh', 'api', f'repos/{a.repo}/tarball/{commit}'], cwd=a.checkout, stdout=archive, check=True, timeout=120)
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode='r:gz') as tar:
            for member in tar:
                if not member.isfile() or '/' not in member.name:
                    continue
                relative = member.name.split('/', 1)[1]
                if relative in expected:
                    seen[relative] = hashlib.file_digest(tar.extractfile(member), 'sha256').hexdigest()
if seen != expected:
    raise RuntimeError('GitHub file set or content identity differs from the pinned local Git snapshot')
release = json.loads(command(['gh', 'api', f'repos/{a.repo}/releases/tags/{a.tag}']))
manifest = json.loads(command(['git', 'show', commit + ':' + prefix + 'results/pilot_v1/checkpoint_manifest.json']))
assets = {r['name']: r for r in release['assets']}
verified = []
for row in manifest:
    remote = assets[row['file']]
    if remote['size'] != row['bytes'] or remote.get('digest') != 'sha256:' + row['sha256'] or remote['state'] != 'uploaded':
        raise RuntimeError('GitHub release asset size, state or server-computed digest mismatch')
    verified.append({'name': row['file'], 'size': remote['size'], 'sha256': row['sha256'],
        'verification': 'GitHub server-computed digest and size matched local/server checkpoint',
        'url': remote['browser_download_url']})
result = {'status': 'GITHUB_PINNED_SOURCE_AND_RELEASE_DIGESTS_VERIFIED',
    'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_commit': commit, 'repo': a.repo, 'source_files_verified': len(expected),
    'verification_method': a.method,
    'source_digest_algorithm': 'Git object SHA1' if a.method == 'git-tree' else 'SHA256 of downloaded archive members',
    'source_file_digests': seen, 'assets': verified, 'release_url': release['html_url'],
    'raw_training_datasets_redistributed': False,
    'scope': 'Pinned tracked subtree content identities and server-computed release SHA256 digests; receipt is outside its own verified snapshot.'}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'source_file_digests'}, ensure_ascii=False))

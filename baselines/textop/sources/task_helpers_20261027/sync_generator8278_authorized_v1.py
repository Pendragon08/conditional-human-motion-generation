"""Publish only this turn's explicit public files; preserve all other work."""
from pathlib import Path
import datetime, hashlib, json, os, shutil, subprocess, msvcrt

TASK = Path(__file__).resolve().parent
REPO = TASK.parent / 'phygile_reproduction_20261017' / 'conditional-human-motion-generation'
BASE = REPO / 'baselines/textop'
DEST = Path('D:/具身智能/TextOp复现_20261024')
BRANCH = 'codex/textop-reproduction-20261024'
names = [
    'CONTINUATION.md', 'README.md', 'SERVER_NOTES.md',
    'GENERATOR8278_CORPUS_ENVIRONMENT.md', 'GENERATOR8278_VAE_ENVIRONMENT.md',
    'configs/generator8278-corpus-spec.json',
    'configs/generator8278-priority-plan.json', 'configs/generator8278-vae-spec.json',
    'provider_tools/collect_generator8278.py',
    'provider_tools/collect_generator8278_report.py',
    'reports/generator8278_training_20261027.md',
    'reports/generator8278_current_20261027.md',
    'scripts/build_generator8278_corpus.py', 'scripts/supervise_generator8278_corpus.py',
    'scripts/generator8278_vae_runtime.py', 'scripts/run_generator8278_vae.py',
    'scripts/supervise_generator8278_vae.py',
]
result_root = BASE / 'results/generator8278_20261027'
names += sorted(p.relative_to(BASE).as_posix() for p in result_root.rglob('*') if p.is_file())
assert len(names) == len(set(names))
assert all((BASE / n).is_file() for n in names)
assert all((BASE / n).suffix in {'.py', '.md', '.json', '.txt'} for n in names)
assert all('.private.' not in n and '_private.' not in n for n in names)
assert not any('checkpoint_sync_state' in n for n in names)

guard = (TASK / 'publish.lock').open('a+b')
if guard.seek(0, 2) == 0:
    guard.write(b'0'); guard.flush()
guard.seek(0)
msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
env = os.environ.copy()
env.update(HTTPS_PROXY='http://127.0.0.1:7897', HTTP_PROXY='http://127.0.0.1:7897', GIT_TERMINAL_PROMPT='0')

def git(*args, stdin=None):
    return subprocess.run(['git', '-C', str(REPO), '-c', 'core.quotepath=false', *args],
        input=stdin, capture_output=True, text=True, encoding='utf-8', check=True,
        timeout=40, env=env).stdout.strip()

assert git('branch', '--show-current') == BRANCH
assert not git('diff', '--cached', '--name-only'), 'Existing staged work; do not mix commits'
allow = {'baselines/textop/' + n for n in names}
git('--literal-pathspecs', 'add', '-f', '--pathspec-from-file=-', '--pathspec-file-nul',
    stdin='\0'.join(sorted(allow)) + '\0')
staged = set(git('diff', '--cached', '--name-only').splitlines())
assert staged and staged <= allow
commit_output = git('commit', '-m', 'Start pre-filter 69D VAE training on available motion corpus')
print(commit_output.splitlines()[0], flush=True)
git('push', 'origin', BRANCH)
head = git('rev-parse', 'HEAD')
remote = git('ls-remote', 'origin', 'refs/heads/' + BRANCH).split()[0]
assert head == remote
receipt = {'branch': BRANCH, 'commit': head, 'verified_remote_head': remote,
    'url': 'https://github.com/Pendragon08/conditional-human-motion-generation/tree/' + BRANCH + '/baselines/textop'}
(BASE / 'provider_tools/github_sync.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')

records = []
for n in names:
    src = BASE / n
    dst = DEST / n if n.startswith('results/') else DEST / 'code/generator8278_20261027' / n
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    assert hashlib.sha256(dst.read_bytes()).hexdigest() == src_sha
    records.append({'source': n, 'destination': str(dst), 'bytes': src.stat().st_size, 'sha256': src_sha})
record = dict(receipt, timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    files=records, private_arrays_or_checkpoints_published=False,
    unrelated_checkpoint_sync_state_included=False)
out = DEST / 'GitHub同步记录_全来源生成器训练_20261027.json'
out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'commit': head, 'remote_verified': True, 'd_files_verified': len(records),
    'record': str(out)}, ensure_ascii=False), flush=True)

"""Read-only process observation and bounded public export; never launches jobs."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/paired_feedback_validation_v51_20261012'
stamp = time.time_ns()
receipt = json.loads((OUT / 'launcher_receipt.json').read_text())
pid = int(receipt['pid'])
process = subprocess.run(['ps', '-p', str(pid), '-o', 'pid,stat,etime,args'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
expected = str(ROOT / 'scripts/run_paired_feedback_validation_v51_20261012.py')
live = process.returncode == 0 and expected in process.stdout and '--run' in process.stdout
observation = dict(time=time.time(), pid=pid, process_query_returncode=process.returncode,
    process_query_stdout=process.stdout, process_query_stderr=process.stderr,
    expected_process_confirmed_live=live,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    role='Point-in-time observation only; does not prove later liveness or efficacy',
    launch_or_restart_calls=0, innovation_confirmed=False, overleaf_results_sync=False)
for name in ('state.json', 'progress.json', 'summary.json', 'budget_ledger.json'):
    file = OUT / name
    if file.exists():
        blob = file.read_bytes()
        try:
            observation[name] = json.loads(blob)
            observation[name + '_sha256'] = hashlib.sha256(blob).hexdigest()
        except ValueError as error:
            observation[name + '_read_error'] = str(error)
path = OUT / ('process_snapshot_%d.json' % stamp)
path.write_text(json.dumps(observation, indent=2), encoding='utf8')
export_dir = OUT / 'public_exports'
export_dir.mkdir(exist_ok=True)
archive = export_dir / ('paired_feedback_v51_public_%d.tar.gz' % stamp)
entries = []
with tarfile.open(archive, 'x:gz') as tar:
    for file in sorted(OUT.rglob('*')):
        if not file.is_file() or file.suffix not in ('.json', '.log', '.md', '.txt'):
            continue
        blob = file.read_bytes()
        if file.suffix == '.json':
            try:
                json.loads(blob)
            except ValueError:
                continue  # live in-progress write, not represented as completed evidence
        relative = str(file.relative_to(ROOT)).replace('\\', '/')
        item = tarfile.TarInfo(relative)
        item.size = len(blob)
        tar.addfile(item, io.BytesIO(blob))
        entries.append(dict(path=relative, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()))
    manifest = json.dumps(dict(snapshot=str(path.relative_to(ROOT)), entries=entries,
                              excluded='All NPZ/model/binary assets; live logs are snapshots'), indent=2).encode()
    item = tarfile.TarInfo('results/paired_feedback_validation_v51_20261012/public_export_%d.json' % stamp)
    item.size = len(manifest)
    tar.addfile(item, io.BytesIO(manifest))
print(json.dumps(dict(archive=str(archive), snapshot=str(path), pid=pid,
    live=live, state=observation.get('state.json'), entries=len(entries))), flush=True)

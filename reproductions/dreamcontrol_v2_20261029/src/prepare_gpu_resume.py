"""Prepare an isolated GPU continuation from a stopped, audited CPU prefix.

The original queue must have terminated; this script copies and validates its
saved prefix, then generates an unlaunched continuation. GPU availability and
CUDA compatibility must be checked separately immediately before starting it.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1: raise ValueError('unexpected solver template fragment: ' + old)
    return text.replace(old, new)


def generate_solver(text):
    text = replace_once(text,
        "    if any(d.platform != 'cpu' for d in jax.devices()):\n        raise ValueError('CPU required')",
        "    devices = jax.devices()\n    if len(devices) != 1 or devices[0].platform != 'gpu':\n        raise ValueError('one explicitly selected GPU required')")
    old = "    with log.open('x', encoding='utf8', buffering=1) as stream:\n        for wi, start in enumerate(starts):"
    new = """    prefix = [json.loads(line) for line in Path(__file__).with_name('prefix.jsonl').read_text().splitlines()]
    with log.open('x', encoding='utf8', buffering=1) as stream:
        for row in prefix:
            stream.write(json.dumps(row, allow_nan=False) + '\\n')
            rows.append(row)
        if prefix:
            with np.load(Path(__file__).with_name(prefix[-1]['artifact']), allow_pickle=False) as f:
                previous = {'indices': f['sample_indices'].copy(), 'root': f['root_wxyz_xyz'].copy(), 'q': f['joint_pos'].copy()}
        stream.flush()
        os.fsync(stream.fileno())
        for wi in range(len(prefix), len(starts)):
            start = starts[wi]"""
    text = replace_once(text, old, new)
    text = replace_once(text, "    report = {'source_sha256': expected,", "    report = {'backend': 'gpu', 'resumed_prefix_windows': len(prefix), 'source_sha256': expected,")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('original_run', 'source', 'urdf', 'prefix_auditor', 'output'):
        ap.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    a = ap.parse_args()
    old = a.original_run.resolve(strict=True)
    status = (old / 'queue_status.txt').read_text().splitlines()
    if not status or status[-1] != 'solver_failed':
        raise ValueError('original queue has not terminated unsuccessfully; do not duplicate a live or complete run')
    a.output.mkdir(parents=False, exist_ok=False)
    prefix_audit = a.output / 'prefix_audit.json'
    subprocess.run([sys.executable, str(a.prefix_auditor), '--run-dir', str(old), '--source', str(a.source),
                    '--urdf', str(a.urdf), '--output', str(prefix_audit)], check=True)
    audit = json.loads(prefix_audit.read_text())
    if audit['prefix_windows'] >= audit['expected_windows']:
        raise ValueError('no remaining windows')
    source_log = old / 'windows.jsonl'; log_hash = sha(source_log)
    if log_hash != audit['prefix_log_sha256']: raise RuntimeError('prefix changed after audit')
    rows = [json.loads(line) for line in source_log.read_text().splitlines()]
    for row in rows:
        original = (old / row['artifact']).resolve(strict=True)
        if not original.is_relative_to(old) or sha(original) != row['artifact_sha256']:
            raise ValueError('prefix artifact mismatch')
        target = a.output / row['artifact']
        with original.open('rb') as src, target.open('xb') as dst: shutil.copyfileobj(src, dst)
        if sha(target) != row['artifact_sha256']: raise RuntimeError('copied artifact mismatch')
    if sha(source_log) != log_hash or (old / 'queue_status.txt').read_text().splitlines()[-1] != 'solver_failed':
        raise RuntimeError('original run changed during staging')
    with (a.output / 'prefix.jsonl').open('x', encoding='utf8') as f:
        for row in rows: f.write(json.dumps(row, allow_nan=False) + '\n')
    solver = generate_solver((old / 'run.py').read_text())
    compile(solver, str(a.output / 'run.py'), 'exec')
    with (a.output / 'run.py').open('x', encoding='utf8', newline='\n') as f: f.write(solver)
    result = {'original_run': str(old), 'original_solver_sha256': sha(old / 'run.py'),
              'original_prefix_sha256': log_hash, 'prefix_audit_sha256': sha(prefix_audit),
              'staged_prefix_sha256': sha(a.output / 'prefix.jsonl'),
              'gpu_solver_sha256': sha(a.output / 'run.py'), 'prefix_windows': len(rows),
              'expected_windows': audit['expected_windows'], 'launched': False,
              'limitations': ['Requires separate GPU process and CUDA compatibility checks.',
                              'CPU prefix and GPU continuation may differ numerically.',
                              'Source, timing, caption, contact and physical quality remain unaccepted.']}
    with (a.output / 'resume_manifest.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()

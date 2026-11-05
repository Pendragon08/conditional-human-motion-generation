"""Prepare an isolated GPU test of the first saved CPU PyRoki window."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OLD = ROOT / 'mosh_000798_pyroki_fullrate_20261031'
HERE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def change(text, a, b):
    if text.count(a) != 1: raise ValueError('unexpected source template')
    return text.replace(a, b)


def main():
    receipt = json.loads((OLD / 'prepare_manifest.json').read_text())
    if sha(OLD / 'run.py') != receipt['solver_sha256']: raise ValueError('source solver changed')
    rows = [json.loads(line) for line in (OLD / 'windows.jsonl').read_text().splitlines()]
    if not rows or rows[0]['window'] != 0 or sha(OLD / rows[0]['artifact']) != rows[0]['artifact_sha256']:
        raise ValueError('first CPU window unavailable or changed')
    text = (OLD / 'run.py').read_text()
    text = change(text, "    if any(d.platform != 'cpu' for d in jax.devices()):\n        raise ValueError('CPU required')",
                  "    devices = jax.devices()\n    if len(devices) != 1 or devices[0].platform != 'gpu':\n        raise ValueError('one explicitly selected GPU required')")
    text = change(text, "    if starts[-1] != len(human) - 8:\n        starts.append(len(human) - 8)",
                  "    if starts[-1] != len(human) - 8:\n        starts.append(len(human) - 8)\n    starts = starts[:1]")
    compile(text, str(HERE / 'run.py'), 'exec')
    with (HERE / 'run.py').open('x', encoding='utf8', newline='\n') as f: f.write(text)
    report = {'source_solver_sha256': sha(OLD / 'run.py'), 'cpu_window_sha256': rows[0]['artifact_sha256'],
              'gpu_solver_sha256': sha(HERE / 'run.py'), 'launched': False,
              'limitations': ['One window is a smoke comparison, not a full numerical equivalence or physical acceptance.']}
    with (HERE / 'prepare_manifest.json').open('x') as f: json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()

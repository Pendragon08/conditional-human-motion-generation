"""Prepare isolated CPU PyRoki window run from a verified MoSh reconstruction.

The existing first-generation optimizer and assumed floor are approximations;
results and any failed run remain non-training evidence.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HUMAN = ROOT / 'mosh_000798_fullrate_20261031'
TEMPLATE = ROOT / 'accad_004347_pyroki_fullrate_20261031'
HERE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1: raise ValueError('unexpected template occurrence: ' + old)
    return text.replace(old, new)


def main():
    human_report = HUMAN / 'output/human_report.json'
    human_audit = HUMAN / 'human_audit.json'
    human = json.loads(human_report.read_text()); checked = json.loads(human_audit.read_text())
    source = HUMAN / 'output/human_120hz.npz'
    template_report = json.loads((TEMPLATE / 'report.json').read_text())
    if (sha(human_report) != checked['human_report_sha256'] or sha(source) != human['artifact_sha256'] or
        sha(source) != checked['artifact_sha256'] or human['frames'] != 1015 or
        sha(TEMPLATE / 'run.py') != template_report['script_sha256']):
        raise ValueError('human or solver template provenance mismatch')
    solver = (TEMPLATE / 'run.py').read_text()
    for old, new in [
        ("accad_004347_fullrate_20261031/human_120hz.npz", "mosh_000798_fullrate_20261031/output/human_120hz.npz"),
        (template_report['source_sha256'], human['artifact_sha256']),
        ("converted.shape != (355, 22, 3)", "converted.shape != (1015, 22, 3)"),
        ('len(times) != 355', 'len(times) != 1015'),
        ('ACCAD candidate path/FPS and official intermediate identity unverified',
         'MoSh candidate path/FPS and official intermediate identity unverified')]:
        solver = replace_once(solver, old, new)
    audit = (TEMPLATE / 'audit.py').read_text()
    for old, new in [
        ("accad_004347_fullrate_20261031/human_120hz.npz", "mosh_000798_fullrate_20261031/output/human_120hz.npz"),
        ("report['window_count'] == 88", "report['window_count'] == 253"),
        ('set(range(1, 355))', 'set(range(1, 1015))')]:
        audit = replace_once(audit, old, new)
    launcher = (TEMPLATE / 'run.sh').read_text()
    launcher = replace_once(launcher, 'accad_004347_pyroki_fullrate_20261031', HERE.name)
    launcher = replace_once(launcher, 'timeout 2400', 'timeout 3600')
    queue = '''#!/usr/bin/env bash
set -uC
cd "$(dirname "$0")"
printf '%s\\n' started > queue_status.txt
if bash run.sh; then
  if env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python audit.py > audit.log 2>&1; then
    printf '%s\\n' audited >> queue_status.txt
  else
    printf '%s\\n' audit_failed >> queue_status.txt
  fi
else
  printf '%s\\n' solver_failed >> queue_status.txt
fi
'''
    for name, content in [('run.py', solver), ('audit.py', audit), ('run.sh', launcher), ('queue.sh', queue)]:
        with (HERE / name).open('x', encoding='utf8', newline='\n') as f: f.write(content)
    receipt = {'human_report_sha256': sha(human_report), 'human_audit_sha256': sha(human_audit),
               'source_sha256': sha(source), 'template_solver_sha256': sha(TEMPLATE / 'run.py'),
               'solver_sha256': sha(HERE / 'run.py'), 'audit_sha256': sha(HERE / 'audit.py'),
               'launcher_sha256': sha(HERE / 'run.sh'), 'queue_sha256': sha(HERE / 'queue.sh'),
               'expected_windows': 253, 'source_frames': 1015, 'training_ready': False}
    with (HERE / 'prepare_manifest.json').open('x', encoding='utf8') as f:
        json.dump(receipt, f, indent=2); f.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__': main()

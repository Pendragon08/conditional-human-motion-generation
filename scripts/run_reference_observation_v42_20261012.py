"""Two validation-only tracing batches; no candidate efficacy or score changes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import run_reset_alignment_checks_v4 as runtime
from verify_execution_lineage_v39_20261012 import guard_batch, sha, read, ANCHORS
from physics_guided_com_lineage_v24_20261012 import request

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/condition_feedback_formal_v28_20261012'
OUT = ROOT / 'results/reference_observation_v42_20261012'


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf8')


def prepare():
    for name, digest in ANCHORS.items():
        assert sha(BASE / name) == digest
    version = read(BASE / 'versions.json')
    assert all(sha(ROOT / name) == digest for name, digest in version['files'].items())
    for name in ('scripts/reference_observation_collect_v42_20261012.py',
                 'scripts/run_reference_observation_v42_20261012.py',
                 'scripts/verify_execution_lineage_v39_20261012.py',
                 'scripts/compare_reference_observation_v42_20261012.py',
                 'docs/reference_observation_v42_20261012.md'):
        version['files'][name] = sha(ROOT / name)
    version.update(role='Validation-only passive reference/observation interface diagnosis',
                   source_manifest_sha256=sha(BASE / 'source_manifest.json'),
                   thresholds_sha256=sha(BASE / 'frozen_thresholds.json'),
                   simulation_batches=2, candidate_efficacy=False, overleaf_results_sync=False)
    if OUT.exists():
        assert read(OUT / 'versions.json') == version
        return
    OUT.mkdir()
    for name in ('source_manifest.json','frozen_thresholds.json'):
        shutil.copyfile(BASE / name, OUT / name)
    save(OUT / 'versions.json', version)
    save(OUT / 'budget_ledger.json', [])
    (OUT / 'requests').mkdir()
    (OUT / 'raw').mkdir()


def batch(label, mode, replay_file=None):
    expected = request(ROOT, OUT, 'val', mode, 11, 0,
                       {'replay_file':replay_file} if replay_file else {})
    path = OUT / 'raw' / label
    wanted = OUT / 'requests' / (label + '.json')
    ledger = read(OUT / 'budget_ledger.json')
    if path.exists():
        assert wanted.exists() and read(wanted) == expected
        guard_batch(path, expected)
        assert sha(path / 'observation_trace.npz') == read(path / 'metadata.json')['observation_trace_sha256']
        assert any(row['label']==label and row['exit_code']==0 for row in ledger)
        return path
    # Include every previously recorded stage plus this new directory.
    latest = sorted((ROOT / 'results/round_resource_status_20261012').glob('snapshot_*.json'))[-1]
    previous = read(latest)
    directories = [ROOT / name for name in previous['directories']]
    if OUT not in directories:
        directories.append(OUT)
    all_rows = [row for d in directories if (d/'budget_ledger.json').exists()
                for row in read(d/'budget_ledger.json')]
    spent = sum(row['seconds'] for row in all_rows)
    measurement = sum(row['seconds'] for row in all_rows if row['stage']=='measurement')
    timeout = int(min(600, 72*3600-spent, 4*3600-measurement))
    assert timeout >= 60
    assert shutil.disk_usage('/data').free >= 20*1024**3
    assert sum(p.stat().st_size for d in directories for p in d.rglob('*') if p.is_file()) < 15*1024**3
    runtime.OUT=OUT
    resources=runtime.resource_guard()
    save(wanted, expected)
    command=[str(runtime.PYTHON),'-u',str(ROOT/'scripts/reference_observation_collect_v42_20261012.py'),
             '--manifest',str(OUT/'source_manifest.json'),'--split','val','--mode',mode,
             '--seed','11','--output',str(path),'--thresholds',str(OUT/'frozen_thresholds.json'),
             '--request-file',str(wanted)]
    if replay_file:
        command += ['--replay-file',str(replay_file)]
    start=time.monotonic()
    code=-1
    save(OUT/'state.json',dict(status='RUNNING',label=label,resources=resources,time=time.time()))
    try:
        with (OUT/(label+'.log')).open('x') as log:
            result=subprocess.run(command,cwd=ROOT/'third_party/CLoSD',env=runtime.environment(),
                                  stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        code=result.returncode
    finally:
        ledger.append(dict(label=label,stage='measurement',seconds=time.monotonic()-start,
                           exit_code=code,request_sha256=sha(wanted)))
        save(OUT/'budget_ledger.json',ledger)
    assert code==0, 'Failed attempt preserved; no silent retry'
    guard_batch(path, expected)
    assert sha(path/'observation_trace.npz')==read(path/'metadata.json')['observation_trace_sha256']
    return path


def main():
    prepare()
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            first=batch('val_planned_s11','planned')
            import numpy as np
            with np.load(first/'trajectories.npz',allow_pickle=False) as arrays:
                meta=read(first/'metadata.json')
                if not (OUT/'raw_reference.npz').exists():
                    np.savez_compressed(OUT/'raw_reference.npz',reference=arrays['reference'],
                                        original_goal=np.asarray(meta['original_goal'],np.float32))
            replay=batch('val_replay_s11','replay',OUT/'raw_reference.npz')
            from compare_reference_observation_v42_20261012 import compare
            summary=compare(first,replay)
            save(OUT/'summary.json',summary)
            save(OUT/'state.json',dict(status='COMPLETE_INTERFACE_DIAGNOSTIC',time=time.time()))
            print(json.dumps(summary,indent=2),flush=True)
        except Exception as error:
            save(OUT/'state.json',dict(status='FAILED_RETAINED',error=repr(error),time=time.time()))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args()
    main()

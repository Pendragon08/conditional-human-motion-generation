"""Single native-contact availability probe; no retry or failed-output overwrite."""
import json
from pathlib import Path
import shutil
import subprocess
import time
import run_reset_alignment_checks_v4 as runtime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/articulated_contact_fixture_v22_20261012'


def save(path, value):
    path.write_text(json.dumps(value, indent=2))


def main():
    import fcntl
    OUT.mkdir(exist_ok=True)
    with (OUT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dest = OUT / 'first_attempt'
        assert not dest.exists(), 'Single attempt only; retain any previous failure'
        directories = [ROOT / path for path in json.loads((ROOT / 'results/contact_measurement_review_20261011/budget_snapshot_20261012.json').read_text())['directories']]
        previous = [row for path in directories if (path / 'budget_ledger.json').exists() for row in json.loads((path / 'budget_ledger.json').read_text())]
        remaining = min(120, 72*3600-sum(x['seconds'] for x in previous),
                        4*3600-sum(x['seconds'] for x in previous if x['stage']=='measurement'))
        assert remaining > 60
        assert shutil.disk_usage('/data').free >= 20*1024**3
        assert sum(p.stat().st_size for d in directories+[OUT] for p in d.rglob('*') if p.is_file()) < 15*1024**3
        files = ['scripts/articulated_contact_fixture_v22_20261012.py','scripts/run_articulated_contact_fixture_v22_20261012.py',
                 'scripts/run_reset_alignment_checks_v4.py','scripts/physics_guided_metric_v21_20261012.py','scripts/native_contact_diagnostic.py',
                 'configs/env-closd-baseline-20261010.json','docs/articulated_contact_fixture_v22_20261012.md']
        hashes = {p:runtime.sha(ROOT/p) for p in files}
        save(OUT/'versions.json', hashes)
        runtime.OUT = OUT
        resources = runtime.resource_guard()
        save(OUT/'state.json', dict(status='RUNNING', resources=resources))
        begin = time.monotonic()
        with (OUT/'first_attempt.log').open('w') as log:
            try:
                code = subprocess.run([str(runtime.PYTHON), '-u', str(ROOT/'scripts/articulated_contact_fixture_v22_20261012.py'), '--output', str(dest)],
                    env=runtime.environment(), stdout=log, stderr=subprocess.STDOUT, timeout=remaining).returncode
            except subprocess.TimeoutExpired:
                code=124
        save(OUT/'budget_ledger.json', [dict(stage='measurement',label='first_attempt',seconds=time.monotonic()-begin,exit_code=code)])
        if code:
            save(OUT/'state.json',dict(status='FAILED_RETAINED',exit_code=code))
            raise RuntimeError('Native API attempt failure preserved')
        summary = json.loads((dest/'summary.json').read_text())
        assert runtime.sha(dest/'fixture_arrays.npz') == summary['arrays_sha256']
        assert summary['source_sha256'] == hashes['scripts/articulated_contact_fixture_v22_20261012.py']
        save(OUT/'state.json',dict(status='COMPLETE_DIAGNOSTIC',velocity_calibration_status=summary['status']))
        print('ARTICULATED_CONTACT_DOCUMENT_EXECUTION_COMPLETE',summary['status'],flush=True)


if __name__ == '__main__':
    main()

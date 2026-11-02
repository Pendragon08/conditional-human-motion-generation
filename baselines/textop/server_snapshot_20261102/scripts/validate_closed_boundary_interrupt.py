"""CPU-only newly-owned fixture; never targets a trainer or imports Torch."""
import argparse, hashlib, json, os, pathlib, shutil, subprocess, sys, time
from closed_boundary_interrupt import identity, interrupt_at_guard, require_identity

p = argparse.ArgumentParser(); p.add_argument('--run', required=True); a = p.parse_args()
ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
assert a.run.isascii() and a.run.isidentifier() and os.getuid() == 1013
out = ROOT / 'pause_control_fixtures' / a.run; out.mkdir(parents=True, exist_ok=False)
fixture = out / 'fixture.py'
fixture.write_text('''import json,pathlib,subprocess,time,sys
out=pathlib.Path(sys.argv[1]); boundary=False; step=0
try:
 while step<20000:
  boundary=False
  step+=4
  time.sleep(.001)
  boundary=True
  subprocess.check_output(['findmnt','-T',str(out),'-n','-o','FSTYPE'],text=True)
except KeyboardInterrupt:
 if boundary:
  fs=subprocess.check_output(['findmnt','-T',str(out),'-n','-o','FSTYPE'],text=True).strip()
  (out/'simulated_emergency.json').write_text(json.dumps(dict(boundary=boundary,step=step,fs=fs,simulation_only=True)))
  sys.exit(41)
 sys.exit(42)
''', encoding='utf-8')
log = (out / 'fixture.log').open('xb')
child = subprocess.Popen([sys.executable, str(fixture), str(out)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
checks = []
try:
    time.sleep(.1)
    expected = identity(child.pid)
    changed = dict(expected, start_ticks=expected['start_ticks'] + 1)
    try: require_identity(expected, changed)
    except AssertionError: checks.append('start_tick_mismatch_rejected')
    else: raise AssertionError('identity mismatch accepted')
    result = interrupt_at_guard(expected, ['findmnt', '-T', str(out), '-n', '-o', 'FSTYPE'],
        str(pathlib.Path(shutil.which('findmnt')).resolve()), out / 'control.jsonl', 30, 30)
    rc = child.wait(timeout=30); checks.append('actual_owned_fixture_exit_observed')
    saved = json.loads((out / 'simulated_emergency.json').read_text())
    assert rc == 41 and saved['boundary'] and saved['step'] % 4 == 0
    checks.append('interruption_caught_only_at_closed_boundary')
    journal = [json.loads(x) for x in (out / 'control.jsonl').read_text().splitlines()]
    ints = [x for x in journal if x['event'] == 'signal_sent' and x['signal'] == 'SIGINT']
    assert len(ints) == 1 and ints[0]['pid'] == child.pid
    checks.append('exactly_one_sigint_to_original_fixture_pidfd')
    assert not any(x.get('signal') in ('SIGTERM', 'SIGKILL') for x in journal)
    checks.append('no_forced_termination')
    assert set(x['pid'] for x in journal if x['event'] == 'signal_sent') <= {child.pid, next(x['child']['pid'] for x in journal if x['event']=='storage_guard_held')}
    checks.append('signals_only_fixture_and_its_one_findmnt_child')
    receipt = {'status': 'CPU_BOUNDARY_INTERRUPT_FIXTURE_PASSED', 'actual_fixture_exit': rc,
        'checks': checks, 'fixture_saved': saved, 'control_result': result,
        'gpu_runs': 0, 'training_signals': 0, 'real_training_checkpoints_created': 0,
        'source_sha256': {n: hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest() for n in
            ['closed_boundary_interrupt.py','validate_closed_boundary_interrupt.py','quick_pidfd.py']}}
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))
finally:
    if child.poll() is None: child.terminate(); child.wait(timeout=5)
    log.close()

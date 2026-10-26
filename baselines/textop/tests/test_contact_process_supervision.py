"""Real CPU subprocess/failure tests, not SDK or robot validation. Preserve all files."""
import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from contact_process_supervision import run_supervised


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=pathlib.Path, required=True)
    a = p.parse_args()
    root = a.output.absolute()
    assert os.name == 'posix' and not root.exists()
    root.mkdir()
    os.sched_setaffinity(0, {0, 1, 2, 3})
    checks = []
    sentinel = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(90)'],
                                start_new_session=True)
    try:
        def case(name, source, timeout=3, fail_sample=None):
            out = root / name
            out.mkdir()
            child = out / 'child.py'
            child.write_text(source)
            calls = [0]

            def sample(pid):
                calls[0] += 1
                if calls[0] == fail_sample:
                    raise RuntimeError('Synthetic sampler failure')
                return {'synthetic_resource_sample': True, 'owned_child_pid': pid}

            result = run_supervised([sys.executable, '-u', str(child)], out / 'stdout.log',
                                    out / 'samples.jsonl', out / 'proof.json', sample,
                                    timeout, sample_interval=0.05, grace_seconds=0.2)
            assert sentinel.poll() is None, 'Unrelated owned sentinel was affected'
            assert not result.get('cleanup_error')
            assert result == json.loads((out / 'proof.json').read_text())
            assert result['samples_written'] == len((out / 'samples.jsonl').read_text().splitlines())
            if result['child_pid'] is not None:
                assert result['child_reaped'] is True
                assert all(x['owned_group'] == result['child_pid'] for x in result['shutdown_actions'])
            checks.append({'case': name, 'proof': str(out / 'proof.json'),
                           'supervision_passed': result['supervision_passed'],
                           'actual_child_exit': result['child_exit_code'],
                           'sentinel_untouched': True})
            return result, out

        r, out = case('natural_exit', 'import time;time.sleep(.25);print("cpu-only complete")')
        assert r['supervision_passed'] and r['child_exit_code'] == 0 and r['samples_written'] >= 4
        r, out = case('nonzero_exit', 'raise SystemExit(7)')
        assert not r['supervision_passed'] and r['child_exit_code'] == 7 and not r['timed_out']
        r, out = case('timeout_term', 'import time;time.sleep(90)', timeout=.4)
        assert not r['supervision_passed'] and r['timed_out'] and r['child_exit_code'] == -signal.SIGTERM
        r, out = case('timeout_kill',
                      'import signal,time,pathlib\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
                      'pathlib.Path(__file__).with_name("partial_trials.jsonl").write_text("retained failure fixture\\n")\n'
                      'time.sleep(90)', timeout=.6)
        assert r['timed_out'] and r['child_exit_code'] == -signal.SIGKILL
        assert [x['signal'] for x in r['shutdown_actions']] == ['SIGTERM', 'SIGKILL']
        assert (out / 'partial_trials.jsonl').read_text() == 'retained failure fixture\n'
        r, out = case('sampler_failure_running', 'import time;time.sleep(90)', fail_sample=3)
        assert not r['supervision_passed'] and 'Synthetic sampler failure' in r['supervisor_error']
        assert r['child_exit_code'] == -signal.SIGTERM and not r['timed_out']
        r, out = case('sampler_failure_before_launch', 'raise AssertionError("must not run")', fail_sample=1)
        assert not r['supervision_passed'] and r['child_pid'] is None and not r['child_reaped']

        # Send SIGTERM to a separately owned supervisor; it must reap only its child.
        out = root / 'supervisor_sigterm'
        out.mkdir()
        driver = out / 'driver.py'
        driver.write_text('import sys,pathlib,json\n'
            'sys.path.insert(0,' + repr(str(pathlib.Path(__file__).resolve().parents[1] / 'scripts')) + ')\n'
            'from contact_process_supervision import run_supervised\n'
            'out=pathlib.Path(__file__).parent\n'
            'def sample(pid):\n'
            ' if pid is not None:(out/"ready").write_text(str(pid))\n'
            ' return {"synthetic_resource_sample":True}\n'
            'r=run_supervised([sys.executable,"-c","import time;time.sleep(90)"],'
            'out/"child.log",out/"samples.jsonl",out/"proof.json",sample,30,.05,.2)\n'
            'raise SystemExit(0 if r["supervision_passed"] else 9)\n')
        with (out / 'driver.log').open('xb') as stream:
            proc = subprocess.Popen([sys.executable, str(driver)], stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 5
            while not (out / 'ready').exists() and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(.01)
            assert (out / 'ready').exists(), 'CPU supervisor failed to start'
            proc.send_signal(signal.SIGTERM)
            assert proc.wait(timeout=5) == 9
        r = json.loads((out / 'proof.json').read_text())
        assert not r['supervision_passed'] and r['child_reaped'] and not r['timed_out']
        assert 'SupervisorSignal' in r['supervisor_error'] and r['child_exit_code'] == -signal.SIGTERM
        assert sentinel.poll() is None
        checks.append({'case': 'supervisor_sigterm', 'proof': str(out / 'proof.json'),
                       'supervision_passed': False, 'sentinel_untouched': True,
                       'actual_child_exit': r['child_exit_code']})
    finally:
        if sentinel.poll() is None:
            sentinel.terminate()
        sentinel.wait(timeout=5)
    receipt = {'status': 'CONTACT_SUPERVISION_CPU_PROCESS_TESTS_PASSED', 'checks': checks,
               'checks_passed': len(checks), 'sdk_calls': 0, 'gpu_jobs': 0, 'physics_steps': 0,
               'optimizer_updates': 0, 'actual_cpu_subprocesses': True,
               'physical_filter_executed': False, 'dataset_accepted': False,
               'files_removed': False, 'cpu_affinity': sorted(os.sched_getaffinity(0))}
    (root / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

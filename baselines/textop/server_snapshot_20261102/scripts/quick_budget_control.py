"""Observe by default; stop only this run after its5k checkpoint is fully backed up.

No active trainer code/config is edited. The original supervisor's actual exit
and interruption records are preserved, never relabeled as a completed40k run.
"""
import argparse
import hashlib
import json
import os
import pathlib
import signal
import time
from quick_budget_guards import ROOT, RUN, OUT, validate_progress, validate_group, validate_backup
from quick_pidfd import BoundProcess, start_ticks

root = pathlib.Path(ROOT)
p = argparse.ArgumentParser()
p.add_argument('--action', choices=['observe', 'finish'], default='observe')
a = p.parse_args()
assert os.getuid() == 1013
os.sched_setaffinity(0, {0, 1, 2, 3})
plan_path = root / 'configs/quick-reproduction-plan-20261026.json'
plan = json.loads(plan_path.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert plan['requested_run'] == RUN and plan['quality_label'] == 'QUICK_REPRODUCTION_NOT_PAPER_COMPLETE'
for name, value in plan['frozen_runtime_files'].items():
    assert sha(root / name) == value, 'Active runtime must remain unchanged'
for name, value in plan['control_sources'].items():
    assert sha(root / name) == value
receipt_path = pathlib.Path(OUT) / 'receipt.json'
checkpoint = pathlib.Path(OUT) / 'completed_5000.pt'
launch_path = root / 'receipts' / ('filter_policy_' + RUN + '_launch.json')
exit_path = root / 'receipts' / ('filter_policy_' + RUN + '_exit.json')
stop_path = root / 'receipts/quick5000_controlled_stop.json'
pending = root / 'receipts/quick5000_stop_requested.json'

def complete(request):
    original_exit = json.loads(exit_path.read_text())
    assert original_exit['child_pid'] == request['owned_group']
    assert sha(checkpoint) == request['checkpoint']['sha256']
    result = dict(request, status='USER_AUTHORIZED_QUICK_5000_EARLY_STOP_COMPLETE',
                  original_supervisor_exit=original_exit,
                  original_exit_receipt_sha256=sha(exit_path), original_final_receipt_sha256=sha(receipt_path),
                  full_paper_training_completed=False, quick_checkpoint_completed_iterations=5000,
                  transient_iterations_after_checkpoint_not_recoverable=True,
                  checkpoint_retained=True, timestamp_unix=time.time())
    with stop_path.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))

if stop_path.exists():
    print(stop_path.read_text())
    raise SystemExit(0)
if pending.exists():
    request = json.loads(pending.read_text())
    if a.action == 'observe':
        print(json.dumps(dict(request, status='QUICK_STOP_PENDING', original_exit_observed=exit_path.exists())))
        raise SystemExit(0)
    assert exit_path.exists(), 'Prior signal attempt still pending; never send a duplicate signal'
    backup = json.loads((root / 'receipts/quick5000_backup_verified.json').read_text())
    validate_backup(backup, request['checkpoint'])
    complete(request)
    raise SystemExit(0)
r = json.loads(receipt_path.read_text())
reached = validate_progress(r, plan)
launch = json.loads(launch_path.read_text())
members = []
for proc in pathlib.Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        if os.getpgid(int(proc.name)) != launch['child_pid']:
            continue
        members.append({'pid': int(proc.name), 'uid': proc.stat().st_uid,
                        'group': os.getpgid(int(proc.name)), 'session': os.getsid(int(proc.name)),
                        'start_ticks': start_ticks(int(proc.name)),
                        'python_executable': (proc / 'exe').resolve().name.startswith('python'),
                        'argv': (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')})
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
trainer, group = validate_group(launch, members)
ready = reached and checkpoint.exists() and time.time() - checkpoint.stat().st_mtime > 30
metadata = None
if ready:
    latest = r['latest_checkpoint']
    assert latest['name'] == checkpoint.name and latest['completed_learning_iterations'] == 5000
    assert latest['actual_optimizer_updates'] == 100000
    metadata = {'path': str(checkpoint), 'bytes': checkpoint.stat().st_size, 'sha256': sha(checkpoint)}
    assert metadata['sha256'] == latest['sha256'] and metadata['bytes'] == latest['bytes']
observation = {'status': 'QUICK_BUDGET_TARGET_READY' if ready else 'QUICK_BUDGET_WAITING_FOR_5000',
               'quick_target_iterations': 5000, 'current_completed_iterations': r['completed_learning_iterations'],
               'optimizer_updates': r['optimizer_updates'], 'target_checkpoint_ready': ready,
               'checkpoint': metadata, 'trainer_pid': trainer, 'owned_group': group,
               'enabled': plan['early_stop_enabled'], 'quality_label': plan['quality_label'],
               'signals_sent': [], 'sdk_runs': 0, 'data_deleted': False, 'sample_unix': time.time()}
if a.action == 'observe':
    print(json.dumps(observation))
    raise SystemExit(0)
assert plan['early_stop_enabled'] is True and ready, 'Never interrupt before the verified target checkpoint'
assert not exit_path.exists(), 'Existing exit requires investigation, not another signal'
backup_path = root / 'receipts/quick5000_backup_verified.json'
backup = json.loads(backup_path.read_text())
validate_backup(backup, metadata)
identity = next(m for m in members if m['pid'] == trainer)
handle = BoundProcess(trainer, identity['start_ticks'])
assert os.getpgid(trainer) == group and pathlib.Path('/proc/' + str(trainer)).stat().st_uid == 1013
assert not pending.exists(), 'Preserve prior stop attempt; inspect before any retry'
request = dict(observation, action='user_authorized_early_stop_after_backup',
               plan_sha256=sha(plan_path), backup_receipt_sha256=sha(backup_path),
               user_request=plan['user_request'], requested_signal='SIGINT',
               trainer_start_ticks=identity['start_ticks'], signal_identity='pidfd_original_process',
               original_40000_completed=False, original_runtime_modified=False)
with pending.open('x') as f:
    json.dump(request, f, indent=2)
signals = []
def sent(entry):
    signals.append(dict(entry, timestamp_unix=time.time()))
    request['signals_sent'] = list(signals)
    temporary = pending.with_suffix('.partial')
    temporary.write_text(json.dumps(request, indent=2));temporary.replace(pending)

# SIGINT gives the trainer its existing exception/finally path to flush writers.
handle.send(signal.SIGINT)
sent({'pid': trainer, 'signal': 'SIGINT'})
deadline = time.monotonic() + 35
while time.monotonic() < deadline and not exit_path.exists():
    time.sleep(.25)
if not exit_path.exists():
    if handle.alive():
        handle.send(signal.SIGTERM)
        sent({'pid': trainer, 'signal': 'SIGTERM'})
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not exit_path.exists():
        time.sleep(.25)
if not exit_path.exists():
    if handle.alive():
        handle.send(signal.SIGKILL)
        sent({'pid': trainer, 'signal': 'SIGKILL'})
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not exit_path.exists():
        time.sleep(.25)
assert exit_path.exists(), 'Keep pending signal evidence; original supervisor exit not observed'
handle.close()
complete(request)

"""Stop only the existing supported PPO queue on explicit user cancellation."""
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
J = ROOT / 'stage3_supported_references_20261104/task_ppo_20261104'
active_path = J / 'active_controller.json'
active = json.loads(active_path.read_text())
pid = active['pid']
assert pid == 1573490
out = J / 'user_cancelled_after_pour_20261105_v1'
out.mkdir(exist_ok=False)

def info(p):
    base = Path('/proc') / str(p)
    try:
        fields = (base / 'stat').read_text().split(') ', 1)[1].split()
        return {'pid': p, 'ppid': int(fields[1]), 'start_ticks': fields[19], 'state': fields[0], 'command': (base / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'), 'uid': base.stat().st_uid}
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None

root_info = info(pid)
assert root_info and root_info['uid'] == os.getuid()
assert str(ROOT / 'stage3_supported_references_20261104/src/stage3_supported_ppo_queue_v1.py') in root_info['command']
for p in [active_path, J / 'queue_v1/status.json']:
    (out / ('before_' + p.name)).write_bytes(p.read_bytes())
os.kill(pid, signal.SIGSTOP)
stopped = []
try:
    processes = {int(p.name): info(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
    selected = {pid}
    while True:
        children = {p for p, item in processes.items() if item and item['ppid'] in selected}
        if children.issubset(selected):
            break
        selected.update(children)
    before = [processes[p] for p in selected if processes.get(p)]
    (out / 'processes_before.json').write_text(json.dumps(before, indent=2))
    for old in sorted(before, key=lambda r: r['pid'] == pid):
        current = info(old['pid'])
        if current and current['start_ticks'] == old['start_ticks']:
            assert current['uid'] == os.getuid()
            os.kill(old['pid'], signal.SIGTERM)
            stopped.append(old)
finally:
    current = info(pid)
    if current and current['start_ticks'] == root_info['start_ticks']:
        os.kill(pid, signal.SIGCONT)
deadline = time.monotonic() + 8
while time.monotonic() < deadline:
    alive = [old for old in stopped if (current := info(old['pid'])) and current['start_ticks'] == old['start_ticks'] and current['state'] != 'Z']
    if not alive:
        break
    time.sleep(.2)
else:
    for old in alive:
        os.kill(old['pid'], signal.SIGKILL)
    time.sleep(.5)
alive = [old for old in stopped if (current := info(old['pid'])) and current['start_ticks'] == old['start_ticks'] and current['state'] != 'Z']
assert not alive
receipt = {'status': 'cancelled_by_user_after_pour_completion', 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'reason': 'User explicitly requested stopping subsequent skill PPO training after Pick/Pour zero-success evaluations.', 'controller_pid': pid, 'stopped_processes': stopped, 'remaining_same_identity_processes': [], 'completed_pick_and_pour_models_and_evaluations_preserved': True, 'bimanual_box_lift_and_box_push_pull_formal_training_cancelled': True, 'original_controller_status_preserved': str(out / 'before_status.json'), 'new_diffusion_repair_and_github_upload_untouched': True}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2))
active.update(status='cancelled_by_user', active=False, stopped_utc=receipt['utc'], cancellation_receipt=str(out / 'receipt.json'), status_path=str(out / 'receipt.json'))
temporary = active_path.with_suffix('.json.tmp')
temporary.write_text(json.dumps(active, indent=2))
temporary.replace(active_path)
print(json.dumps(receipt))

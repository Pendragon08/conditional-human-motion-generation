"""Reconcile completed reference metadata with existing live controllers, never launch jobs."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path

R = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
B = R / 'stage3_supported_references_20261104'
D = R / 'stage3_supermimic29_20261103'
N = D / 'native_universal_20261104'
Q = R / 'stage3_skill_sequence_20261103'
F = B / 'mixed2000_final_v1'
now = datetime.datetime.now(datetime.timezone.utc).isoformat()

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def atomic(p, value):
    temp = p.with_name(p.name + '.reference2000_metadata_tmp')
    with temp.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, p)

receipt = read(F / 'completion_receipt.json')
assert receipt['status'] == 'four_task2000_reference_dataset_completed'
assert receipt['per_task'] == dict.fromkeys(['pick', 'pour', 'bimanual_box_lift', 'box_push_pull'], 500)
for key in ['manifest', 'native_loader_report', 'bulk_completion']:
    assert sha(Path(receipt[key])) == receipt[key + '_sha256']
guard = read(N / 'formal2000_sequence_v1/status.json')
assert guard['status'] != 'failed_preserved' and guard['manifest_sha256'] == receipt['manifest_sha256']
assert guard['status'] in ['early600', 'main', 'evaluation', 'native2000_training_and_body_tracking_evaluation_completed']
ppo_path = B / 'task_ppo_20261104/pour/ppo_8912_5000_v1/status.json'
ppo = read(ppo_path)
assert ppo['policy_updates'] > 0 and ppo['reference_count'] == 500
snapshot = {'checked_utc': now, 'reference_count': 2000, 'per_task': receipt['per_task'],
            'completion_receipt': str(F / 'completion_receipt.json'),
            'completion_receipt_sha256': sha(F / 'completion_receipt.json'),
            'native_manifest_sha256': receipt['manifest_sha256'],
            'native_controller_status': str(N / 'formal2000_sequence_v1/status.json'),
            'native_phase': guard['status'], 'native_process_started': True,
            'native_checkpoint_verified': bool(guard.get('formal_training_started')),
            'native_early_checkpoint': guard.get('early_checkpoint'),
            'native_metrics_snapshot': guard.get('native_metrics_tail'),
            'pour_status': str(ppo_path), 'pour_updates_snapshot': ppo['policy_updates'],
            'object_task_success_claimed': False}
backup = Q / 'before_reference2000_handoff_20261105'
backup.mkdir(exist_ok=True)
with (Q / 'controller.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    targets = [Q / 'sequence.json', Q / 'progress.json', D / 'adaptation_status.json',
               N / 'all_four_reference_gate.json', B / 'plan.json', B / 'active_pipeline.json']
    for p in targets:
        dest = backup / (str(p.relative_to(R)).replace('/', '__'))
        if not dest.exists():
            dest.open('xb').write(p.read_bytes())
    for p in [Q / 'sequence.json', Q / 'progress.json']:
        data = read(p)
        for task in data['tasks']:
            name = task['id']
            task.update(admitted_references=500, final_library_count=500,
                        mixed_reference_receipt=str(F / 'completion_receipt.json'))
            if name == 'pick':
                task.update(status='training3000_and100env_evaluation_archived_success0_of100',
                            observed_success_ratio=0.0, task_success_achieved=False)
            elif name == 'pour':
                task.update(status='formal_PPO_running', run_dir=str(ppo_path.parent),
                            training_pid=ppo['pid'], policy_updates_snapshot=ppo['policy_updates'],
                            live_progress=str(ppo_path))
            else:
                task.update(status='reference500_completed_waiting_PPO_lane',
                            live_progress=str(B / 'task_ppo_20261104/queue_v1/status.json'))
        data.update(status='four_task2000_ready_native_and_Pour_PPO_running',
                    next_phase='parallel_native_tracker_and_task_PPO_training',
                    next_action='Monitor existing immutable native and PPO controllers through checkpoints and independent final evaluations; no duplicate launches.',
                    last_checked_utc=now, reference2000_handoff_20261105=snapshot)
        if p.name == 'progress.json':
            data.update(current_index=1, current_task='pour')
        else:
            data['version'] = max(data.get('version', 7), 8)
        mode = data.get('SuperMimic_policy_mode', {})
        mode.update(native500_validation_only=False, actual_reference_count=2000,
                    final2000_receipt=str(F / 'completion_receipt.json'),
                    native_status=str(N / 'formal2000_sequence_v1/status.json'))
        data['SuperMimic_policy_mode'] = mode
        policy = data.get('parallel_training_policy_20261104', {})
        policy.update(status='native2000_and_Pour_PPO_processes_running', reference_gate_satisfied=True,
                      formal_training_started=bool(guard.get('formal_training_started')))
        policy.setdefault('native_lane', {}).update(formal_process_started=True,
            formal_run_started=bool(guard.get('formal_training_started')),
            optimizer_checkpoint_verified=bool(guard.get('formal_training_started')))
        policy.setdefault('ppo_lane', {}).update(active_task='pour', formal_training_started=True)
        data['parallel_training_policy_20261104'] = policy
        atomic(p, data)
    data = read(N / 'all_four_reference_gate.json')
    assert data['training_allowed'] and data['manifest_sha256'] == receipt['manifest_sha256']
    data.update(verified_counts=receipt['per_task'], updated_utc=now,
                other_tasks_stage='all_four_task_reference_libraries_and_mixed_native_loader_completed',
                native500_validation_only=False,
                completion_receipt=str(F / 'completion_receipt.json'),
                completion_receipt_sha256=sha(F / 'completion_receipt.json'),
                next_action='Follow existing native controller phase ' + guard['status'] + '; actual optimizer counters, checkpoint proof and all2000 body evaluation remain authoritative.')
    atomic(N / 'all_four_reference_gate.json', data)
    data = read(D / 'adaptation_status.json')
    data.update(status='native2000_process_running', phase=guard['status'], reference_count=2000, per_task=receipt['per_task'],
                formal_process_started=True, formal_training_started=bool(guard.get('formal_training_started')),
                reference2000_handoff_20261105=snapshot, updated_utc=now)
    atomic(D / 'adaptation_status.json', data)
    for p in [B / 'plan.json', B / 'active_pipeline.json']:
        data = read(p)
        data.update(status='four_task2000_reference_dataset_completed', updated_utc=now,
                    final_reference_counts=receipt['per_task'], active_source_jobs=[],
                    completion_receipt=str(F / 'completion_receipt.json'),
                    completion_receipt_sha256=sha(F / 'completion_receipt.json'))
        atomic(p, data)
out = B / 'reference2000_handoff_20261105.json'
atomic(out, snapshot)
print(json.dumps(snapshot))

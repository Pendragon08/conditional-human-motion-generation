"""Read-only provenance for the recovery run's NEW 15k..40k checkpoints.

No Torch import, signal, launch, checkpoint load, output creation or pass flag.
"""
import hashlib
import json
import pathlib
import re
import time

RUN = 'filter_policy_paper69_resume10000_filter_recovery_20261027_v1'
SPEC = 'configs/filter-recovery-backup-spec.json'
RESUME_SPEC = 'configs/teacher10000-filter-recovery-spec.json'


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def decide(spec, receipt, completed, exits=None):
    assert spec['run'] == RUN
    assert type(completed) is int and completed in range(15000, 40001, 5000)
    assert receipt['mode'] == 'formal'
    assert receipt['status'] in ('RUNNING_FORMAL_FILTER_TRAINING', 'TEXTOP_FORMAL_FILTER_TRAINING_FINISHED')
    for key, value in spec['training_receipt_bindings'].items():
        assert receipt[key] == value, key
    assert receipt['num_envs'] == 8192 and receipt['reference_sources'] == 8278
    assert receipt['resume_exact_state_verified'] is True
    assert receipt['target_learning_iterations'] == 40000
    assert receipt['contact_profile'] == spec['contact_profile']
    count, updates = receipt['completed_learning_iterations'], receipt['optimizer_updates']
    assert type(count) is int and type(updates) is int
    meta = receipt.get('latest_checkpoint')
    # The immutable trainer writes 0 until its first resumed 100-round flush.
    # Do not rewrite that receipt or pretend those zeroes describe a checkpoint.
    if count == updates == 0 and meta is None:
        assert receipt['status'] == 'RUNNING_FORMAL_FILTER_TRAINING'
        return {'ready': False, 'status': 'PENDING_FIRST_RESUMED_RECEIPT_FLUSH', 'observed_completed': 0}
    assert 10000 <= count <= 40000 and updates == 20 * count
    if meta is None:
        return {'ready': False, 'status': 'PENDING_NO_NEW_CHECKPOINT', 'observed_completed': count}
    actual = meta['completed_learning_iterations']
    assert type(actual) is int and actual in range(15000, 40001, 5000) and actual <= count
    assert type(meta['actual_optimizer_updates']) is int and meta['actual_optimizer_updates'] == 20 * actual
    assert meta['name'] == 'completed_{}.pt'.format(actual)
    assert type(meta['bytes']) is int and 0 < meta['bytes'] <= 128 * 1024**2
    assert re.fullmatch('[0-9a-f]{64}', meta['sha256'])
    assert actual <= completed, 'Newer checkpoint exists: latest-only provenance cannot be relaxed'
    if actual < completed:
        return {'ready': False, 'status': 'PENDING_REQUESTED_CHECKPOINT', 'latest_completed': actual, 'observed_completed': count}
    if completed == 40000:
        if receipt['status'] != 'TEXTOP_FORMAL_FILTER_TRAINING_FINISHED' or receipt['output_writers_closed'] is not True:
            return {'ready': False, 'status': 'PENDING_FINAL_TRAINING_CLOSE', 'observed_completed': count}
        if not exits or any(exits.get(k) is None for k in ('process', 'supervisor', 'outer')):
            return {'ready': False, 'status': 'PENDING_ACTUAL_FINAL_EXITS', 'observed_completed': count}
        for key in ('process', 'supervisor'):
            e = exits[key]
            assert type(e['child_exit_code']) is int and e['child_exit_code'] == 0
            assert e['child_reaped'] is True and e['supervision_passed'] is True
            assert e['timed_out'] is False and e.get('supervisor_error') is None
        assert type(exits['outer']['actual_returncode']) is int and exits['outer']['actual_returncode'] == 0
    return {'ready': True, 'status': 'METADATA_READY_FILE_CHECKS_REQUIRED', 'observed_completed': count, 'metadata': meta}


def preflight(root, spec, completed, now=None):
    root = pathlib.Path(root)
    assert root.resolve() == root and root.stat().st_uid == 1013
    sp = root / RESUME_SPEC
    assert sha(sp) == spec['resume_spec_sha256']
    resume = json.loads(sp.read_text(encoding='utf-8'))
    assert resume['run'] == spec['run'] == RUN
    assert resume['checkpoint'] == spec['resumed_checkpoint']
    for name, digest in resume['frozen_bindings'].items():
        assert sha(root / name) == digest, name
    assert sha(root / 'scripts/supervise_teacher10000_filter_recovery.py') == resume['source_sha256']
    for name, digest in spec['bound_backup_files'].items():
        assert sha(root / name) == digest, name
    prefix = 'filter_recovery_' + RUN
    lp = root / 'receipts' / (prefix + '_launch.json')
    launch = json.loads(lp.read_text(encoding='utf-8'))
    assert launch['run'] == RUN and launch['spec_sha256'] == spec['resume_spec_sha256']
    assert launch['checkpoint_sha256'] == spec['resumed_checkpoint']['sha256']
    assert launch['resumed_completed'] == 10000 and launch['resumed_updates'] == 200000
    out = root / 'tracker_validation' / RUN
    assert out.resolve() == out and out.stat().st_uid == 1013
    rp = out / 'receipt.json'
    raw = rp.read_bytes()
    receipt = json.loads(raw)
    exits = {}
    for key, suffix in [('process', '_process_exit.json'), ('supervisor', '_exit.json'), ('outer', '_outer_exit.json')]:
        p = root / 'receipts' / (prefix + suffix)
        exits[key] = json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
    decision = decide(spec, receipt, completed, exits)
    decision.update(run=RUN, requested_completed=completed, training_receipt_sha256=hashlib.sha256(raw).hexdigest(),
                    launch_sha256=sha(lp), sdk_dispatches=0, checkpoint_loads=0, export_files_created=0)
    if not decision['ready']:
        return decision
    cp = out / decision['metadata']['name']
    assert cp.resolve() == cp and cp.stat().st_uid == 1013 and cp.is_file()
    before = cp.stat()
    age = (time.time() if now is None else now) - before.st_mtime
    if age <= 30:
        return dict(decision, ready=False, status='PENDING_FILE_STABILITY', file_age_seconds=age)
    assert before.st_size == decision['metadata']['bytes'] and sha(cp) == decision['metadata']['sha256']
    after = cp.stat()
    assert (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
    again = decide(spec, json.loads(rp.read_text(encoding='utf-8')), completed, exits)
    assert again['ready'] and again['metadata'] == decision['metadata'], 'Latest checkpoint changed while hashing'
    return dict(decision, status='FILTER_RECOVERY_CHECKPOINT_READY_FOR_CPU_AUDIT', checkpoint=str(cp), file_age_seconds=age)

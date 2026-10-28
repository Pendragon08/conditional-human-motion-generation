"""Explicit stopped100k LDM provenance; no model imports or production signals.

The old program's KeyboardInterrupt is retained. A controller snapshot may
contain post-target memory counters: the immutable checkpoint is authoritative.
"""
import hashlib
import json
import os
import pathlib
from filtered15000_vae_completion import ROOT, sha, read, owned, closed_job
from ldm100k_control_protocol import checkpoint_ready, verify_checkpoint

KIND = 'USER_REQUESTED_LDM100000_PREFIX'
STATUS = 'USER_LDM100000_STOP_ACCEPTED_ORIGINAL_EXITS_RETAINED'
DATA_KEYS = ['corpus_receipt_sha256', 'data_acceptance_sha256',
             'strict_audit_receipt_sha256', 'vae_checkpoint_sha256',
             'vae_inference_sha256']


def validate_records(control, policy, before, final):
    """Pure metadata contract, also exercised with isolated negative fixtures."""
    assert policy['control_authorized'] and policy['target_primitive_attempts'] == 100000
    assert control['status'] == STATUS and control['signals_sent'] == 1
    assert control['output_writers_closed'] and control['common_lock_released']
    assert control['selected_checkpoint_attempts'] == 100000
    assert control['original300k_completion_claimed'] is False
    assert control['original_run'] == policy['original_run']
    assert before['status'] == 'RUNNING_FILTERED15000_LDM_TRAINING'
    assert final['status'] == 'FAILED_FILTERED15000_LDM'
    assert final['error'] == 'KeyboardInterrupt()'
    assert before['mode'] == final['mode'] == 'train'
    assert before['quality_baseline_manager_step'] == 0
    assert before['completed_primitive_attempts'] >= 100000
    ck = control['selected_checkpoint']
    assert ck['manager_step'] == 100000 and ck['path'] == policy['checkpoint_path']
    assert 0 < ck['actual_optimizer_updates'] <= 100000
    assert sum(x == ck for x in before['checkpoints']) == 1
    assert sum(x == ck for x in final['checkpoints']) == 1
    assert checkpoint_ready(before, policy) == ck
    for k, sk in [('spec_sha256', 'original_spec_sha256'),
                  ('source_sha256', 'original_entry_sha256'),
                  ('runtime_binding_sha256', 'original_runtime_binding_sha256')]:
        assert before[k] == final[k] == policy[sk]
    for k in DATA_KEYS:
        assert before[k] == final[k] == policy[k]
    return ck


def expected_state_binding(policy, initial_receipt):
    return dict(stage='ldm', feature_dim=69, dof=29, quick_training=False,
                formal_training=True, paper_schedule=True, prefilter_data=False,
                actual_strict_input=True, paper_two_stage_route_reproduced=False,
                quality_label=initial_receipt['quality_label'],
                numeric_sha256=policy['original_runtime_binding_sha256'],
                corpus_receipt_sha256=policy['corpus_receipt_sha256'],
                vae_checkpoint_sha256=policy['vae_checkpoint_sha256'],
                vae_inference_sha256=policy['vae_inference_sha256'])


def validate_state_metadata(state, selected, expected_binding):
    assert state['schema'] == 'textop.generator_training_state.v1'
    assert state['step'] == selected['manager_step'] == 100000
    assert state['binding'] == expected_binding
    assert state['checkpoint_boundary'] == 'completed_four_primitive_batch; no cached batch/history'
    assert state['ema_models'] == {}
    clock = state['primitive_clock']
    assert clock['completed_primitive_attempts'] == 100000
    assert clock['actual_optimizer_updates'] == selected['actual_optimizer_updates']
    assert clock['last_observed_updates'] == clock['actual_optimizer_updates']
    assert clock['skipped_updates'] >= 0
    assert clock['actual_optimizer_updates'] + clock['skipped_updates'] == 100000
    fields = state['manager_fields']
    # Native evaluation at100k is drained before the original file is saved.
    assert fields['pending_evaluation_steps'] <= 0
    assert fields['eval_loss_totals'] == fields['eval_extra_totals'] == {}
    assert fields['stage_idx'] in [1, 2]
    extra = fields['extra']
    counters = {}
    for k in ['full_sampling_history_calls', 'selected_rollout_history_primitives']:
        assert type(extra[k]) is int and 0 <= extra[k] <= 100000
        counters[k] = extra[k]
    # Zero counters are recorded as findings after the stop, never permission
    # for extra training. The fixture probe is not actual later-stage training.
    findings = [k + '_is_zero' for k, value in counters.items() if value == 0]
    return dict(primitive_clock=clock, actual_checkpoint_history_counters=counters,
                history_findings=findings, saved_native_stage=fields['stage_idx'],
                saved_pending_evaluation_steps=fields['pending_evaluation_steps'])


def stopped_ldm100k(binding):
    assert binding['kind'] == KIND and binding['selected_step'] == 100000
    run = binding['control_run']
    assert run == 'user_ldm100k_control_v1'
    cp = owned(ROOT / 'ldm100k_user_handoff_20261028' / run / 'receipt.json')
    assert sha(cp) == binding['control_receipt_sha256']
    control = read(cp)
    sp = owned(ROOT / 'configs/ldm100k-user-control-spec.json')
    policy = read(sp)
    assert control['spec_sha256'] == sha(sp) == binding['control_spec_sha256']
    assert control['source_sha256'] == policy['entrypoint_sha256']
    assert sha(ROOT / 'scripts/watch_ldm100k_handoff.py') == policy['entrypoint_sha256']
    assert sha(ROOT / 'scripts/supervise_ldm100k_handoff.py') == policy['supervisor_sha256']
    for n, h in policy['bound_sources'].items():
        assert sha(ROOT / n) == h, n
    closure = closed_job('ldm100k_control_' + run, cp, sha(cp), policy['supervisor_sha256'])
    assert closure['exit_sha256'] == binding['control_exit_sha256']
    source_spec = owned(ROOT / 'configs/user-vae100k-ldm-spec.json')
    spec = read(source_spec)
    assert sha(source_spec) == policy['original_spec_sha256']
    assert spec['ldm']['stages'] == [50000, 50000, 50000, 150000]
    numeric_keys = ['ldm', 'seed', 'corpus_receipt_sha256', 'data_acceptance_sha256',
                    'strict_audit_receipt_sha256', 'vae_training', 'vae_export',
                    'vae_inference', 'bound_sources', 'entrypoint_sha256',
                    'checkpoint_dir', 'checkpoint_allowed_root', 'checkpoint_policy',
                    'checkpoint_layout']
    signature = hashlib.sha256(json.dumps({k: spec[k] for k in numeric_keys},
                              sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert signature == policy['original_runtime_binding_sha256']
    rp = owned(ROOT / 'filtered15000_20261027/ldm_runs' / policy['original_run'] / 'receipt.json')
    assert sha(rp) == control['original_final_receipt_sha256']
    final = read(rp)
    before = read(owned(cp.parent / 'producer_before_signal.json'))
    assert read(owned(cp.parent / 'producer_final_snapshot.json')) == final
    selected = validate_records(control, policy, before, final)
    prefix = 'filtered15000_ldm_' + policy['original_run']
    for key, suffix in [('exit', '_exit.json'), ('process', '_process_exit.json'),
                        ('outer', '_outer_exit.json')]:
        path = owned(ROOT / 'receipts' / (prefix + suffix))
        assert sha(path) == control['original_exit_sha256'][key]
        actual = read(path)
        assert actual == control['original_exit_records'][key]
        if key == 'outer':
            assert actual['actual_returncode'] != 0
        else:
            assert actual['child_reaped'] and actual['child_exit_code'] in [-2, 1, 130]
            assert not actual['timed_out'] and actual['supervisor_error'] is None
            assert actual['sample_errors'] == [] and not actual.get('cleanup_error')
    names = [str(ROOT / 'scripts' / n) for n in
             ['run_user_vae100k_ldm.py', 'supervise_user_vae100k_ldm.py',
              'watch_ldm100k_handoff.py', 'supervise_ldm100k_handoff.py']]
    for proc in pathlib.Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != 1013:
                continue
            argv = (proc / 'cmdline').read_bytes().decode().split('\0')
            assert not any(n in argv for n in names), 'Original/control worker still present'
        except (FileNotFoundError, ProcessLookupError):
            pass
    verify_checkpoint(selected, policy['checkpoint_allowed_root'])
    return dict(policy=policy, source_spec=spec, control=control, closure=closure,
                checkpoint=selected, initial_receipt=before,
                expected_state_binding=expected_state_binding(policy, before),
                original_final_receipt_sha256=sha(rp), control_receipt_sha256=sha(cp))

"""Isolated metadata negatives, never real checkpoints/models or production PIDs."""
import argparse
import copy
import json
import pathlib
import sys

B = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B / 'scripts'))
from ldm100k_prefix_completion import validate_records, validate_state_metadata, expected_state_binding
from ldm100k_control_protocol import RESTORE_KEYS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = pathlib.Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    policy = dict(control_authorized=True, target_primitive_attempts=100000,
                  original_run='fixture_only', original_spec_sha256='a' * 64,
                  original_entry_sha256='b' * 64, original_runtime_binding_sha256='c' * 64,
                  checkpoint_path='/fixture/ldm_state_100000.pth', checkpoint_upper_bound_bytes=314572800,
                  corpus_receipt_sha256='d' * 64, data_acceptance_sha256='e' * 64,
                  strict_audit_receipt_sha256='f' * 64, vae_checkpoint_sha256='1' * 64,
                  vae_inference_sha256='2' * 64)
    ck = dict(path=policy['checkpoint_path'], bytes=123, sha256='3' * 64,
              manager_step=100000, actual_optimizer_updates=100000)
    before = dict(status='RUNNING_FILTERED15000_LDM_TRAINING', mode='train', quality_baseline_manager_step=0,
                  completed_primitive_attempts=100004, optimizer_updates=100004, skipped_updates=0,
                  checkpoints=[ck], latest_file_resume_verification={k: True for k in RESTORE_KEYS},
                  spec_sha256='a' * 64, source_sha256='b' * 64, runtime_binding_sha256='c' * 64,
                  corpus_receipt_sha256='d' * 64, data_acceptance_sha256='e' * 64,
                  strict_audit_receipt_sha256='f' * 64, vae_checkpoint_sha256='1' * 64,
                  vae_inference_sha256='2' * 64, vae_pretraining_attempts=100000,
                  full_sampling_history_calls=50004, selected_rollout_history_primitives=20002,
                  quality_label='fixture_no_weights')
    final = dict(before, status='FAILED_FILTERED15000_LDM', error='KeyboardInterrupt()')
    control = dict(status='USER_LDM100000_STOP_ACCEPTED_ORIGINAL_EXITS_RETAINED', signals_sent=1,
                   output_writers_closed=True, common_lock_released=True, selected_checkpoint_attempts=100000,
                   original300k_completion_claimed=False, original_run='fixture_only', selected_checkpoint=ck)
    assert validate_records(control, policy, before, final) == ck
    expected = expected_state_binding(policy, before)
    state = dict(schema='textop.generator_training_state.v1', step=100000, binding=expected,
                 checkpoint_boundary='completed_four_primitive_batch; no cached batch/history', ema_models={},
                 primitive_clock=dict(completed_primitive_attempts=100000, actual_optimizer_updates=100000,
                                      last_observed_updates=100000, skipped_updates=0),
                 manager_fields=dict(pending_evaluation_steps=0, eval_loss_totals={}, eval_extra_totals={},
                                     stage_idx=1, extra=dict(full_sampling_history_calls=50000,
                                                            selected_rollout_history_primitives=20000)))
    audit = validate_state_metadata(state, ck, expected)
    assert audit['actual_checkpoint_history_counters']['full_sampling_history_calls'] == 50000
    assert before['full_sampling_history_calls'] == 50004  # post-target memory never replaces saved counters
    zero = copy.deepcopy(state)
    zero['manager_fields']['extra'] = dict(full_sampling_history_calls=0, selected_rollout_history_primitives=0)
    assert len(validate_state_metadata(zero, ck, expected)['history_findings']) == 2
    cases = []
    for part, key, value in [('control', 'signals_sent', 0), ('control', 'signals_sent', 2),
                             ('control', 'common_lock_released', False),
                             ('control', 'output_writers_closed', False),
                             ('control', 'original300k_completion_claimed', True),
                             ('control', 'selected_checkpoint_attempts', 300000),
                             ('policy', 'control_authorized', False),
                             ('before', 'status', 'FAILED_FILTERED15000_LDM'),
                             ('before', 'spec_sha256', '0' * 64),
                             ('before', 'vae_checkpoint_sha256', '0' * 64),
                             ('final', 'error', 'capacity_failure'),
                             ('final', 'status', 'FILTERED15000_LDM_300000_TRAINING_FINISHED')]:
        q = copy.deepcopy(dict(control=control, policy=policy, before=before, final=final))
        q[part][key] = value
        cases.append(lambda q=q: validate_records(q['control'], q['policy'], q['before'], q['final']))
    for path, value in [(('step',), 99700), (('binding', 'numeric_sha256'), '0' * 64),
                        (('primitive_clock', 'completed_primitive_attempts'), 300000),
                        (('primitive_clock', 'skipped_updates'), 1),
                        (('manager_fields', 'pending_evaluation_steps'), 4),
                        (('manager_fields', 'eval_loss_totals'), {'partial': 1}),
                        (('manager_fields', 'extra', 'full_sampling_history_calls'), -1),
                        (('manager_fields', 'extra', 'selected_rollout_history_primitives'), True)]:
        q = copy.deepcopy(state)
        item = q
        for k in path[:-1]:
            item = item[k]
        item[path[-1]] = value
        cases.append(lambda q=q: validate_state_metadata(q, ck, expected))
    for call in cases:
        try:
            call()
        except (AssertionError, KeyError):
            pass
        else:
            raise AssertionError('Corrupted isolated metadata accepted')
    result = dict(status='LDM100K_PREFIX_ISOLATED_METADATA_FIXTURES_PASSED', rejected_cases=len(cases),
                  post_target_memory_counters_not_selected=True, zero_history_is_a_recorded_finding=True,
                  actual_production_checkpoint_reads=0, actual_models_loaded=0, actual_gpu_contexts=0,
                  actual_production_signals=0, not_actual_endpoint_acceptance=True)
    (out / 'SUMMARY.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

"""Metadata/counter fixtures ONLY. Never load real checkpoints or signal jobs."""
import argparse
import copy
import json
import pathlib
from filter_recovery_backup_protocol import RUN, SPEC, decide
from filter_recovery_statistics import summarize


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); a = p.parse_args()
    target = pathlib.Path(a.output); assert not target.exists()
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = json.loads((root / SPEC).read_text(encoding='utf-8'))
    checks = []
    receipt = dict(spec['training_receipt_bindings'], mode='formal', status='RUNNING_FORMAL_FILTER_TRAINING',
        num_envs=8192, reference_sources=8278, resume_exact_state_verified=True, target_learning_iterations=40000,
        contact_profile=spec['contact_profile'], completed_learning_iterations=15000, optimizer_updates=300000,
        output_writers_closed=False, latest_checkpoint=dict(completed_learning_iterations=15000,
            actual_optimizer_updates=300000, name='completed_15000.pt', bytes=90000000, sha256='0' * 64))
    def check(name, value):
        assert value, name; checks.append(name)
    def rejects(name, fn):
        try:
            fn()
        except (AssertionError, ValueError, TypeError, KeyError):
            checks.append(name); return
        raise AssertionError('Unexpected acceptance: ' + name)
    check('synthetic_valid_15000_metadata_only', decide(spec, receipt, 15000)['ready'])
    for key, value in [('mode', 'pilot'), ('status', 'FAILED'), ('num_envs', 8), ('reference_sources', 64),
                       ('resume_exact_state_verified', False), ('target_learning_iterations', 10000),
                       ('completed_learning_iterations', 9999), ('optimizer_updates', 300001),
                       ('optimizer_updates', 300000.0), ('completed_learning_iterations', 15000.0)]:
        r = copy.deepcopy(receipt); r[key] = value
        rejects('reject_' + key + '_' + str(value), lambda r=r: decide(spec, r, 15000))
    for key in spec['training_receipt_bindings']:
        r = copy.deepcopy(receipt); r[key] = 'incorrect'
        rejects('reject_binding_' + key, lambda r=r: decide(spec, r, 15000))
    r = copy.deepcopy(receipt); r['contact_profile']['top_z'] = .1
    rejects('reject_changed_contact', lambda: decide(spec, r, 15000))
    wrong = copy.deepcopy(spec); wrong['run'] = 'filter_policy_paper69_resume5000_to40000_20261026_v1'
    rejects('reject_old_run', lambda: decide(wrong, receipt, 15000))
    for n in [10000, 15001, True]:
        rejects('reject_requested_' + str(n), lambda n=n: decide(spec, receipt, n))
    for key, value in [('name', '../completed_15000.pt'), ('sha256', 'bad'), ('bytes', 128*1024**2+1),
                       ('bytes', True), ('actual_optimizer_updates', 299999), ('completed_learning_iterations', 10000)]:
        r = copy.deepcopy(receipt); r['latest_checkpoint'][key] = value
        rejects('reject_metadata_' + key, lambda r=r: decide(spec, r, 15000))
    r = copy.deepcopy(receipt); r['completed_learning_iterations'] = 10050; r['optimizer_updates'] = 201000; r.pop('latest_checkpoint')
    check('no_new_checkpoint_pending', not decide(spec, r, 15000)['ready'])
    r['completed_learning_iterations'] = r['optimizer_updates'] = 0
    check('initial_zero_receipt_pending', decide(spec, r, 15000)['status'] == 'PENDING_FIRST_RESUMED_RECEIPT_FLUSH')
    r['optimizer_updates'] = 200000
    rejects('reject_mixed_zero_count', lambda: decide(spec, r, 15000))
    check('future_requested_pending', not decide(spec, receipt, 20000)['ready'])
    r = copy.deepcopy(receipt); r['completed_learning_iterations'] = 20000; r['optimizer_updates'] = 400000
    r['latest_checkpoint'].update(completed_learning_iterations=20000, actual_optimizer_updates=400000, name='completed_20000.pt')
    rejects('reject_stale_requested_checkpoint', lambda: decide(spec, r, 15000))
    final = copy.deepcopy(receipt); final.update(completed_learning_iterations=40000, optimizer_updates=800000)
    final['latest_checkpoint'].update(completed_learning_iterations=40000, actual_optimizer_updates=800000, name='completed_40000.pt')
    check('final_running_pending', not decide(spec, final, 40000)['ready'])
    final.update(status='TEXTOP_FORMAL_FILTER_TRAINING_FINISHED', output_writers_closed=True)
    check('final_missing_exits_pending', not decide(spec, final, 40000)['ready'])
    exited = dict(child_exit_code=0, child_reaped=True, supervision_passed=True, timed_out=False, supervisor_error=None)
    exits = dict(process=copy.deepcopy(exited), supervisor=copy.deepcopy(exited), outer=dict(actual_returncode=0))
    check('synthetic_final_closed_metadata_only', decide(spec, final, 40000, exits)['ready'])
    exits['supervisor']['child_exit_code'] = 1
    rejects('reject_final_supervision_failure', lambda: decide(spec, final, 40000, exits))
    rows = [dict(canonical_source='synthetic/' + str(i), sha256='0'*64, frames=40+i, split='train' if i<2 else 'val') for i in range(4)]
    counters = dict(raw_failed=[1, 2, 0, 1], raw_success=[19, 18, 0, 2], failed_motion_count=[1, 1.001, 1, 0], success_motion_count=[19, 18.999, 1, 0])
    summary, decisions = summarize(rows, counters, expected_count=4)
    check('all_four_categories_retained', all(n == 1 for n in summary['counts'].values()))
    check('threshold_equality_is_candidate', decisions[0]['decision'] == 'eligible_for_strict')
    check('threshold_above_is_not_candidate', decisions[1]['decision'] == 'above_relaxed_threshold')
    check('nonacceptance_remains_explicit', summary['dataset_accepted'] is False and summary['strict_evaluation_completed'] is False)
    check('no_asset_delete_or_new_physics', summary['original_assets_deleted'] == summary['new_physics_steps'] == 0)
    check('raw_attempts_sum', summary['raw_attempts_total'] == 43)
    check('candidate_frames_only', summary['eligible_original_frames'] == 40)
    rejects('reject_wrong_expected_source_count', lambda: summarize(rows, counters))
    for label, key, val in [('nan','failed_motion_count',float('nan')), ('negative','raw_failed',-1), ('fractional','raw_success',.5)]:
        c = copy.deepcopy(counters); c[key][0] = val
        rejects('reject_counter_' + label, lambda c=c: summarize(rows, c, 4))
    r = copy.deepcopy(rows); r[1]['canonical_source'] = r[0]['canonical_source']
    rejects('reject_duplicate_source', lambda: summarize(r, counters, 4))
    result = dict(status='FILTER_RECOVERY_METADATA_AND_COUNTER_FIXTURES_PASSED', checks=len(checks), names=checks,
        fixtures_only=True, real_new_checkpoint_loads=0, sdk_runs=0, training_signals=0, new_optimizer_updates=0,
        positive_checkpoint_backup_validated=False, physical_filter_completed=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__': main()

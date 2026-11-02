"""CPU gate checks only; synthetic completed counters never stand for real training."""
import argparse
import copy
import json
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from contact_replay_modes import authorize, select_validation_sources, finish_status


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=pathlib.Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    a.output.mkdir()
    spec = dict(formal_evaluation_enabled=False, real_sdk_validation_completed=False,
                validation_launch_enabled=False, validation_source_count=8,
                validation_minimum_frames=50, validation_maximum_frames=500,
                validation_maximum_parallel_envs=8, validation_max_control_calls=1024,
                strict_rollouts_per_motion=1)
    training = dict(status='TEXTOP_FORMAL_FILTER_TRAINING_FINISHED', mode='formal',
                    formal_filter_training_started=True, completed_learning_iterations=40000,
                    optimizer_updates=800000, num_envs=8192, output_writers_closed=True)
    exitproof = dict(child_exit_code=0, timed_out=False, numerical_status=training['status'])
    checks = []

    def rejected(name, mode, mutate):
        s, t, e = copy.deepcopy(spec), copy.deepcopy(training), copy.deepcopy(exitproof)
        mutate(s, t, e)
        try:
            authorize(mode, s, t, e)
        except AssertionError as error:
            checks.append(dict(case=name, expected_rejection=True, reason=str(error)))
        else:
            raise AssertionError(name)

    rejected('formal_disabled', 'formal', lambda s, t, e: None)
    rejected('validation_disabled', 'validation', lambda s, t, e: None)
    rejected('validation_flag_cannot_enable_formal', 'formal', lambda s, t, e: s.update(validation_launch_enabled=True))
    rejected('formal_without_real_sdk_witness', 'formal', lambda s, t, e: s.update(formal_evaluation_enabled=True))
    rejected('validation_incomplete_teacher', 'validation', lambda s, t, e: (
        s.update(validation_launch_enabled=True), t.update(status='RUNNING_FORMAL_FILTER_TRAINING',
                                                         completed_learning_iterations=1000, optimizer_updates=20000)))
    rejected('validation_failed_teacher_exit', 'validation', lambda s, t, e: (
        s.update(validation_launch_enabled=True), e.update(child_exit_code=1)))
    rejected('validation_teacher_not_closed', 'validation', lambda s, t, e: (
        s.update(validation_launch_enabled=True), t.update(output_writers_closed=False)))
    rejected('validation_teacher_timeout', 'validation', lambda s, t, e: (
        s.update(validation_launch_enabled=True), e.update(timed_out=True)))
    enabled = dict(spec, validation_launch_enabled=True)
    assert authorize('validation', enabled, training, exitproof) is True
    assert enabled['formal_evaluation_enabled'] is False and enabled['real_sdk_validation_completed'] is False
    checks.append(dict(case='synthetic_validation_authorization_only', physical_acceptance=False))
    rows = [{'frames': n} for n in [10000, 500, 51, 50, 300, 70, 100, 80, 49, 55, 60, 50]]
    original = copy.deepcopy(rows)
    selection = select_validation_sources(rows, enabled)
    assert selection == [3, 11, 2, 9, 10, 5, 7, 6] and rows == original
    checks.append(dict(case='eight_whole_sources_deterministic_selection', input_arrays_modified=False))
    try:
        select_validation_sources(rows[:8], enabled)
    except AssertionError:
        checks.append(dict(case='insufficient_complete_sources_rejected', expected_rejection=True))
    else:
        raise AssertionError('Insufficient sources accepted')
    assert finish_status('validation') != finish_status('formal')
    checks.append(dict(case='validation_has_separate_nonacceptance_status', passed=True))
    from test_filter_contact_v3 import Fixture
    from apply_tracking_filter_v3 import apply
    fixture = Fixture(a.output / 'synthetic_nonacceptance_fixture')
    fixture.evaluation.update(status=finish_status('validation'), mode='validation',
                              diagnostic_dynamic_rollouts_executed=True, physical_filter_executed=False)
    fixture.eval_exit.update(numerical_status=finish_status('validation'))
    fixture.refresh()
    try:
        apply(fixture.e)
    except AssertionError:
        checks.append(dict(case='validation_receipt_cannot_become_formal_decisions', expected_rejection=True))
    else:
        raise AssertionError('Validation accepted as real physical filter')
    r = dict(status='CONTACT_REPLAY_MODE_CPU_CHECKS_PASSED', checks=checks, checks_passed=len(checks),
             sdk_runs=0, gpu_calls=0, physics_steps=0, optimizer_updates=0,
             physical_filter_executed=False, dataset_accepted=False,
             full_training_simulated_only_in_test_fixture=True)
    (a.output / 'receipt.json').write_text(json.dumps(r, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in r.items() if k != 'checks'}))


if __name__ == '__main__':
    main()

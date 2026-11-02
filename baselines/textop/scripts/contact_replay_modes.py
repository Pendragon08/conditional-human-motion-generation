"""Separate bounded executor validation from full physical-filter evidence.

Both paths require the genuinely completed40k teacher; validation never accepts
motions or emits formal evidence. These functions perform no SDK/GPU work.
"""
from apply_tracking_filter_v3 import require_completed_training

VALIDATION_STATUS = 'TEXTOP_CONTACT_REPLAY_VALIDATION_COMPLETED'
FORMAL_STATUS = 'TEXTOP_TWO_ROUND_FILTER_EXPORT_FINISHED'


def gate_mode(mode, spec):
    assert mode in ['formal', 'validation']
    if mode == 'formal':
        assert spec['formal_evaluation_enabled'] is True, 'Future exporter disabled; no SDK dispatched'
        assert spec['real_sdk_validation_completed'] is True, 'Actual independent SDK validation still required'
    else:
        assert spec['validation_launch_enabled'] is True, 'Future validation disabled; no SDK dispatched'
    return mode == 'validation'


def authorize(mode, spec, training, training_exit):
    diagnostic = gate_mode(mode, spec)
    require_completed_training(training)
    assert training_exit['child_exit_code'] == 0 and training_exit['timed_out'] is False
    assert training_exit['numerical_status'] == training['status']
    return diagnostic


def select_validation_sources(rows, spec):
    count = spec['validation_source_count']
    lower = spec['validation_minimum_frames']
    upper = spec['validation_maximum_frames']
    assert type(count) is int and count == 8
    assert lower == 50 and upper == 500
    assert spec['strict_rollouts_per_motion'] == 1
    assert spec['validation_maximum_parallel_envs'] == count
    assert spec['validation_max_control_calls'] == 1024
    eligible = [i for i, row in enumerate(rows) if lower <= row['frames'] <= upper]
    selected = sorted(eligible, key=lambda i: (rows[i]['frames'], i))[:count]
    assert len(selected) == count, 'Eight complete1–10s original trajectories required'
    # All eight run as one batch. No motion is shortened to fit the budget.
    assert max(rows[i]['frames'] - 1 for i in selected) <= spec['validation_max_control_calls']
    return selected


def finish_status(mode):
    assert mode in ['formal', 'validation']
    return VALIDATION_STATUS if mode == 'validation' else FORMAL_STATUS

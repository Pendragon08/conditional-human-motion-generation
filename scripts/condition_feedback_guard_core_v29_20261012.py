"""C2 proposal only: repeat-agreement feedback and componentwise quality guard.

No simulator, training, or efficacy claim lives in this module. Two repeated
construction/execution calls plus one corrected call consume the same three
candidate calls as the C1 controls. Repeat agreement does not establish that a
systematic error is accurate or that the reference is feasible for all trackers.
"""
import numpy as np

from physics_guided_core import bounded_update, wrap_angle


CONDITION_KEYS = ('endpoint_error_m', 'heading_error_rad', 'endpoint_time_error_s')
PHYSICAL_KEYS = ('surface_slip_proxy_fraction', 'penetration_fraction',
                 'jerk_p95_m_s3')
TRACKING_KEYS = ('tracking_mean_m', 'tracking_p95_m')


def finite_number(row, key):
    value = row.get(key)
    return value is not None and isinstance(value, (int, float)) and np.isfinite(value)


def agreement(first, repeat, thresholds, fraction=0.05):
    """Check repeat agreement, not statistical confidence or physical truth."""
    assert 0 < fraction <= 1
    result = dict(position=False, heading=False, repeat_completed=False)
    if first['case_id'] != repeat['case_id']:
        raise ValueError('Feedback cannot cross source/offset cases')
    if not (first.get('completed', False) and repeat.get('completed', False)):
        return result
    result['repeat_completed'] = True
    first_error = np.asarray(first.get('endpoint_error_vector'), dtype=np.float64)
    repeat_error = np.asarray(repeat.get('endpoint_error_vector'), dtype=np.float64)
    if first_error.shape == repeat_error.shape == (3,) and np.isfinite(first_error).all() and np.isfinite(repeat_error).all():
        result['position'] = bool(np.linalg.norm(first_error-repeat_error) <= fraction*thresholds['goal_position_m'])
    if finite_number(first, 'heading_error_signed_rad') and finite_number(repeat, 'heading_error_signed_rad'):
        delta = wrap_angle(first['heading_error_signed_rad']-repeat['heading_error_signed_rad'])
        result['heading'] = bool(abs(delta) <= fraction*thresholds['heading_rad'])
    return result


def corrected_conditions(metadata, first_rows, repeat_rows, config,
                         use_agreement=True):
    """One bounded update; the independent original evaluation goal is unchanged."""
    assert len(metadata['cases']) == len(first_rows) == len(repeat_rows)
    thresholds = config['base_thresholds']
    reliability = [agreement(a, b, thresholds) for a, b in zip(first_rows, repeat_rows)]
    errors = np.zeros((len(first_rows), 3), dtype=np.float64)
    heading_errors = np.zeros(len(first_rows), dtype=np.float64)
    for i, (a, b, trust) in enumerate(zip(first_rows, repeat_rows, reliability)):
        assert metadata['cases'][i]['case_id'] == a['case_id'] == b['case_id']
        # The no-agreement ablation still spends the repeated call, but uses
        # the first execution's valid endpoint feedback without averaging.
        if use_agreement:
            if trust['position']:
                errors[i] = (np.asarray(a['endpoint_error_vector'])+np.asarray(b['endpoint_error_vector']))/2
            if trust['heading']:
                heading_errors[i] = a['heading_error_signed_rad'] + wrap_angle(b['heading_error_signed_rad']-a['heading_error_signed_rad'])/2
        elif a.get('length', 0) > metadata['context_steps']:
            if np.asarray(a.get('endpoint_error_vector')).shape == (3,) and np.isfinite(a['endpoint_error_vector']).all():
                errors[i] = a['endpoint_error_vector']
            if finite_number(a, 'heading_error_signed_rad'):
                heading_errors[i] = a['heading_error_signed_rad']
    original = np.asarray(metadata['original_goal'], dtype=np.float32)
    proposed = np.asarray(metadata['proposed_goal'], dtype=np.float32)
    goal = bounded_update(original, proposed, errors, config['position_step_m'], config['position_total_m'])
    original_heading = np.asarray(metadata['original_heading'])
    proposed_heading = np.asarray(metadata['proposed_heading'])
    step = np.deg2rad(config['heading_step_deg'])
    total = np.deg2rad(config['heading_total_deg'])
    heading = proposed_heading+np.clip(heading_errors, -step, step)
    heading = original_heading+np.clip(wrap_angle(heading-original_heading), -total, total)
    return dict(proposed_goal=goal.astype(np.float32).tolist(),
                proposed_heading=heading.tolist(), agreement=reliability,
                use_agreement=use_agreement, repeat_agreement_fraction=0.05,
                agreement_is_not_accuracy=True, original_goal_sha256=metadata['original_goal_sha256'])


def acceptable(incumbent, candidate):
    """Reject componentwise deterioration; preserve every source and failure."""
    assert incumbent['case_id'] == candidate['case_id']
    if not candidate.get('completed', False):
        return False, 'candidate_not_complete'
    keys = CONDITION_KEYS+PHYSICAL_KEYS+TRACKING_KEYS
    if not all(finite_number(candidate, key) for key in keys):
        return False, 'candidate_has_missing_or_nonfinite_quality'
    # An incomplete incumbent remains in the record and denominator; a new
    # complete construction can replace it, as in the ordinary C1 selection.
    if not incumbent.get('completed', False):
        return True, 'completed_construction_replaces_incomplete'
    if not all(finite_number(incumbent, key) for key in keys):
        return False, 'incumbent_quality_unverifiable'
    worse = [key for key in keys if candidate[key] > incumbent[key]]
    if worse:
        return False, 'component_regression:'+','.join(worse)
    better = any(candidate[key] < incumbent[key] for key in keys)
    return (True, 'componentwise_nonregression_with_improvement') if better else (False, 'tie_keeps_incumbent')


def choose_guarded(rows_by_candidate):
    """Uses actual evaluated candidates only, with no failed-source filtering."""
    assert len(rows_by_candidate) == 3, 'Two repeats plus one update; no free repeated calls'
    n = len(rows_by_candidate[0])
    assert all(len(rows) == n for rows in rows_by_candidate)
    choices, decisions = [], []
    for i in range(n):
        choice = 0
        case_decisions = []
        for k in (1, 2):
            accept, reason = acceptable(rows_by_candidate[choice][i], rows_by_candidate[k][i])
            case_decisions.append(dict(candidate=k,accepted=accept,reason=reason))
            if accept:
                choice = k
        choices.append(choice)
        decisions.append(case_decisions)
    return choices, decisions

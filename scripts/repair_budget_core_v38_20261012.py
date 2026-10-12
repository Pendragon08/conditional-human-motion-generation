"""C3 prospective allocation core; no efficacy claim or simulator invocation.

Fit repair yield ONLY on validation-source paired attempts. Test outcomes never
enter scheduling. Uniform, random and tracking-difficulty controls share the
same paid call budget and three-call per-input cap. One initial call for every
input guarantees a complete denominator. A scored existing candidate bank is
not a counterfactual acquisition experiment or evidence of compute savings.
"""
from collections import defaultdict
import math
import random


COMPONENTS = ('condition_pass', 'physical_pass', 'tracking_pass')


def signature(row):
    return tuple(bool(row.get(key, False)) for key in COMPONENTS)


def fit_validation(records, validation_sources, forbidden_sources):
    validation_sources, forbidden_sources = set(validation_sources), set(forbidden_sources)
    if validation_sources & forbidden_sources:
        raise ValueError('Source split leakage')
    if not records:
        raise ValueError('No paired validation repair observations')
    bins = defaultdict(lambda: dict(trials=0, repairs=0, calls=0))
    for row in records:
        if row['split'] != 'val' or row['source_id'] not in validation_sources or row['source_id'] in forbidden_sources:
            raise ValueError('Calibration must not use training or test outcomes')
        if row.get('evaluation_phase') != 'fixed_reference_replay':
            raise ValueError('Construction quality is not validated reference replay utility')
        if row['before']['case_id'] != row['after']['case_id']:
            raise ValueError('Unpaired repair calibration')
        calls = row['paid_new_candidate_calls']
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 1:
            raise ValueError('Repair calls must be positively charged')
        if row['before'].get('joint_pass', False):
            raise ValueError('Repair calibration requires initially unqualified input')
        b = bins[signature(row['before'])]
        b['trials'] += 1
        b['calls'] += calls
        b['repairs'] += int(bool(row['after'].get('joint_pass', False)))
    if not any(b['repairs'] for b in bins.values()):
        raise ValueError('No observed joint repair signal; do not invent positive utility')
    return {key: dict(value, expected_joint_repairs_per_paid_call=value['repairs'] / value['calls'])
            for key, value in bins.items()}


def allocate(first_attempts, total_calls, seed, mode, calibration=None, already_paid_calls=None):
    """Return a prospective schedule, not hypothetical outcome or saved cost.

    All inputs first receive one charged observation. Each scheduling step uses
    the initially observed state only; a future sequential runner may update
    states after real paid calls and must log them. No final test score is read.
    """
    if mode not in ('uniform', 'random', 'tracking_difficulty', 'proposed'):
        raise ValueError('Unknown control')
    if mode == 'proposed' and not calibration:
        raise ValueError('Validate repair calibration before allocation')
    if not first_attempts:
        raise ValueError('No inputs')
    case_ids = [r['case_id'] for r in first_attempts]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError('Duplicate input identity')
    n = len(case_ids)
    if isinstance(total_calls, bool) or not isinstance(total_calls, int) or not n <= total_calls <= 3 * n:
        raise ValueError('Budget must cover every source and respect three calls per input')
    paid = [1] * n if already_paid_calls is None else list(already_paid_calls)
    if len(paid) != n or any(type(v) is not int or not 1 <= v <= 3 for v in paid) or sum(paid) > total_calls:
        raise ValueError('Cannot erase existing charged calls')
    rng = random.Random(seed)
    source_counts = defaultdict(int)
    for row, calls in zip(first_attempts, paid):
        source_counts[row['source_id']] += calls
    schedule = []
    while sum(paid) < total_calls:
        eligible = [i for i in range(n) if paid[i] < 3]
        if mode == 'random':
            index = rng.choice(eligible)
        else:
            def score(i):
                row = first_attempts[i]
                if mode == 'uniform':
                    return (paid[i], i)
                if mode == 'tracking_difficulty':
                    value = row.get('tracking_mean_m')
                    difficulty = float(value) if value is not None and math.isfinite(float(value)) else math.inf
                    return (-difficulty, paid[i], i)
                estimate = calibration.get(signature(row))
                # Unseen states have no inferred benefit; exploration is a tie.
                utility = 0. if estimate is None else estimate['expected_joint_repairs_per_paid_call']
                if row.get('joint_pass', False):
                    utility = 0.
                return (-utility, source_counts[row['source_id']], paid[i], i)
            index = min(eligible, key=score)
        paid[index] += 1
        source_counts[first_attempts[index]['source_id']] += 1
        schedule.append(dict(case_id=case_ids[index], source_id=first_attempts[index]['source_id'],
                             paid_candidate_number=paid[index]))
    assert sum(paid) == total_calls and all(1 <= value <= 3 for value in paid)
    return dict(mode=mode, seed=seed, total_paid_calls=total_calls,
                initial_inputs=n, additional_schedule=schedule, per_input_paid_calls=dict(zip(case_ids, paid)),
                outcomes_predicted=False, retrospective_bank_efficacy=False)


def self_check():
    def state(case, condition, physical, tracking, joint=False):
        return dict(case_id=case, condition_pass=condition, physical_pass=physical,
                    tracking_pass=tracking, joint_pass=joint)
    data = [dict(split='val', source_id='v1', evaluation_phase='fixed_reference_replay', before=state('v1a', False, True, True),
                 after=state('v1a', True, True, True, True), paid_new_candidate_calls=1),
            dict(split='val', source_id='v2', evaluation_phase='fixed_reference_replay', before=state('v2a', True, False, True),
                 after=state('v2a', True, False, True), paid_new_candidate_calls=1)]
    calibrated = fit_validation(data, {'v1', 'v2'}, {'t1', 't2'})
    first = [dict(state('t1a', False, True, True), source_id='t1', tracking_mean_m=.05),
             dict(state('t2a', True, False, True), source_id='t2', tracking_mean_m=.2)]
    proposed = allocate(first, 4, 11, 'proposed', calibrated)
    assert proposed['per_input_paid_calls'] == {'t1a': 3, 't2a': 1}
    assert allocate(first, 4, 11, 'uniform')['per_input_paid_calls'] == {'t1a': 2, 't2a': 2}
    assert allocate(first, 4, 11, 'tracking_difficulty')['per_input_paid_calls'] == {'t1a': 1, 't2a': 3}
    assert allocate(first, 4, 11, 'random') == allocate(first, 4, 11, 'random')
    # At the full 3N budget, allocations cannot differ: no allocation novelty.
    for mode in ('uniform', 'random', 'tracking_difficulty', 'proposed'):
        assert set(allocate(first, 6, 11, mode, calibrated)['per_input_paid_calls'].values()) == {3}
    rejected = 0
    for action in (
        lambda: fit_validation([], {'v1'}, set()),
        lambda: fit_validation([data[1]], {'v2'}, set()),
        lambda: fit_validation(data, {'v1', 'v2'}, {'v1'}),
        lambda: fit_validation([dict(data[0], split='test')], {'v1'}, set()),
        lambda: fit_validation([dict(data[0], evaluation_phase='construction')], {'v1'}, set()),
        lambda: allocate(first, 7, 11, 'uniform'),
        lambda: allocate(first, 1, 11, 'uniform'),
        lambda: allocate(first, 4, 11, 'proposed'),
        lambda: allocate(first, 4, 11, 'uniform', already_paid_calls=[3, 3]),
    ):
        try:
            action()
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Invalid scheduling/calibration accepted')
    assert rejected == 9
    print('REPAIR_BUDGET_CORE_FIXTURES_PASS: no simulation efficacy tested')


if __name__ == '__main__':
    self_check()

"""Roundoff-only metric recomputation check; never changes saved scores or gates."""
import math
import sys


def compare_metrics(recomputed, saved):
    differences = []

    def walk(actual, expected, path):
        assert type(actual) is type(expected), 'Metric type changed: ' + path
        if isinstance(actual, dict):
            assert actual.keys() == expected.keys(), 'Metric keys changed: ' + path
            for key in actual:
                walk(actual[key], expected[key], path + '.' + str(key))
        elif isinstance(actual, list):
            assert len(actual) == len(expected), 'Metric length changed: ' + path
            for index, (left, right) in enumerate(zip(actual, expected)):
                walk(left, right, path + '[%d]' % index)
        elif isinstance(actual, float):
            assert math.isfinite(actual) and math.isfinite(expected), 'Nonfinite metric: ' + path
            delta = abs(actual - expected)
            tolerance = 32 * sys.float_info.epsilon * max(1.0, abs(actual), abs(expected))
            assert delta <= tolerance, 'Metric differs beyond float64 roundoff: ' + path
            if delta:
                differences.append(dict(path=path, saved=expected, recomputed=actual,
                    absolute_difference=delta, tolerance=tolerance))
        else:
            # Includes pass/fail flags, identifiers, counters, None and strings.
            assert actual == expected, 'Discrete metric or classification changed: ' + path

    walk(recomputed, saved, 'metrics')
    return dict(status='PASS_ROUNDOFF_ONLY', float64_epsilon=sys.float_info.epsilon,
        tolerance_rule='32 * float64_epsilon * max(1, abs(saved), abs(recomputed))',
        differences=differences, changed_floating_values=len(differences),
        maximum_absolute_difference=max([row['absolute_difference'] for row in differences] or [0]),
        classifications_identical=True, thresholds_changed=False,
        saved_scores_unchanged=True, scope='Numeric recomputation only; lineage verifies other inputs separately.')


if __name__ == '__main__':
    assert compare_metrics([dict(angle=0.07121248986375006, passed=False)],
                           [dict(angle=0.07121248986374962, passed=False)])['changed_floating_values'] == 1
    for left, right in (([dict(passed=True)], [dict(passed=False)]),
                        ([dict(error=0.100001)], [dict(error=0.1)]),
                        ([dict(value=1.0)], [dict(value=1)]),
                        ([dict(value=float('nan'))], [dict(value=float('nan'))])):
        try:
            compare_metrics(left, right)
        except AssertionError:
            continue
        raise RuntimeError('Invalid equivalence accepted')
    print('METRIC_NUMERIC_EQUIVALENCE_CONTRACT_PASS')

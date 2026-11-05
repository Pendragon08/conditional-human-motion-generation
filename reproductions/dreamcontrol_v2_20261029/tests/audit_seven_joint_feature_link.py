"""Cross-check seven linked joint-limit maxima against prior independent audits."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = HERE / 'report.json'
    current = json.loads(path.read_text())
    prior_path = ROOT / 'six_joint_diagnostics_20261031/report.json'
    prior_audit = json.loads((prior_path.parent / 'independent_audit.json').read_text())
    assert sha(prior_path) == prior_audit['input_report_sha256']
    previous = {row['sample_id']: row for row in json.loads(prior_path.read_text())['cases']}
    candidate = json.loads((ROOT / 'accad_004347_fullrate_candidate_20261031/report.json').read_text())
    checked = 0
    for row in current['cases']:
        mid = row['sample_id']
        if mid == 'ACCAD_004347':
            expected = candidate['max_joint_limit_violation_after_filter_rad']
            assert row['feature_sha256'] == candidate['output_sha256']
        else:
            assert row['feature_sha256'] == previous[mid]['feature_sha256']
            expected = previous[mid]['feature_grid_linear_joints']['max_limit_excess_rad']
        assert math.isclose(row['joint_check']['max_excess_rad'], expected, rel_tol=1e-10, abs_tol=1e-10)
        assert row['joint_check']['joint_limits_passed'] == (expected <= 1e-6)
        checked += 1
    assert checked == 7 and current['training_ready'] is False
    result = {'input_report_sha256': sha(path), 'prior_report_sha256': sha(prior_path),
              'checked_samples': checked, 'joint_maxima_agree': True, 'training_ready': False,
              'limitations': ['Prior independent audit recomputed joint maxima; this check does not independently recompute FK or decoded points.',
                              'No source-time, language or physics acceptance is implied.']}
    with (HERE / 'independent_audit.json').open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

"""Compare integrated preflight outcomes with the previous linked-state audit."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    log = HERE / 'preflight.log'
    result = json.loads(log.read_text())
    prior_path = ROOT / 'seven_joint_feature_link_20261031/report.json'
    prior_audit = json.loads((prior_path.parent / 'independent_audit.json').read_text())
    assert sha(prior_path) == prior_audit['input_report_sha256']
    previous = json.loads(prior_path.read_text())
    full_reencoding = json.loads((ROOT / 'seven_full_feature_reencoding_20261031/report.json').read_text())
    full_by_id = {row['sample_id']: row for row in full_reencoding['cases']}
    assert result['manifest_sha256'] == previous['manifest_sha256']
    current = {item['sample_id']: item for item in result['state_audit']['cases']}
    assert set(current) == {item['sample_id'] for item in previous['cases']}
    for old in previous['cases']:
        new = current[old['sample_id']]
        assert new['feature_sha256'] == old['feature_sha256']
        assert new['joint_check'] == old['joint_check']
        assert math.isclose(new['fk_decode_max_error_m'], old['fk_decode_max_error_m'], abs_tol=1e-12)
        assert new['state_sha256'] == full_by_id[old['sample_id']]['state_sha256']
        assert new['feature_reencoding_max_abs_error'] == full_by_id[old['sample_id']]['max_abs_diff']
        assert new['foot_velocity_squared_threshold'] == full_reencoding['foot_velocity_squared_threshold']
    assert result['state_audit']['state_technical_checks_passed'] is False
    assert result['feature_audit']['training_ready'] is False
    output = {'preflight_sha256': sha(log), 'previous_audit_report_sha256': sha(prior_path),
              'cases_checked': len(current), 'results_agree': True, 'training_ready': False,
              'limitations': ['Same FK/decoder/encoder implementation; compares prior results, not independent full re-encoding.',
                              'This does not establish source, time, caption or physical acceptance.']}
    with (HERE / 'independent_audit.json').open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()

"""Check the sampled SFU and ACCAD candidates with a numeric-only tolerance."""
import hashlib
import json
from pathlib import Path
import numpy as np
from g1_joint_limit_gate import check_joint_limits, read_joint_limits, require_joint_limits

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE = Path(__file__).resolve().parent
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    limits = read_joint_limits(URDF)
    records = []
    for name, folder in [('SFU_009033', 'sfu_009033_fullrate_candidate_v2_20261031'),
                         ('ACCAD_004347', 'accad_004347_fullrate_candidate_20261031')]:
        source = ROOT / folder / 'candidate.npz'
        report = json.loads((ROOT / folder / 'report.json').read_text())
        assert sha(source) == report['output_sha256'] and sha(URDF) == report['urdf_sha256']
        with np.load(source, allow_pickle=False) as data:
            q = data['filtered_20hz'][:, 7:].copy()
            names = [str(n) for n in data['joint_names']]
        result = check_joint_limits(q, names, limits, tolerance_rad=1e-6)
        assert abs(result['max_excess_rad'] - report['max_joint_limit_violation_after_filter_rad']) < 1e-12
        if name == 'SFU_009033':
            assert not result['joint_limits_passed']
            try:
                require_joint_limits(q, names, limits, tolerance_rad=1e-6)
            except ValueError:
                pass
            else:
                raise AssertionError('SFU should be rejected')
        else:
            assert result['joint_limits_passed']
        records.append({'sample_id': name, 'candidate_sha256': sha(source), 'check': result})
    result = {'urdf_sha256': sha(URDF), 'cases': records, 'training_ready': False,
              'limitations': ['1e-6 rad is a numeric allowance, not a paper feasibility criterion.',
                              'A pass is limited to saved 20 Hz joint values and does not establish physical quality.',
                              'This isolated gate is not yet integrated into the guarded training entry.']}
    with (HERE / 'sfu_witness.json').open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

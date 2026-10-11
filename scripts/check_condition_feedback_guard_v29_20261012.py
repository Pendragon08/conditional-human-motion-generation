"""Numeric invariants only, using synthetic rows; never counts as sim efficacy."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from condition_feedback_guard_core_v29_20261012 import (
    agreement, acceptable, choose_guarded, corrected_conditions)

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT/'configs/physics_guided_20261011.json').read_text())
thresholds = config['base_thresholds']
base = dict(case_id='synthetic:0', completed=True, length=120,
            endpoint_error_vector=[0.2, 0., 0.], heading_error_signed_rad=0.2,
            endpoint_error_m=0.2, heading_error_rad=0.2, endpoint_time_error_s=0.,
            surface_slip_proxy_fraction=0.3, penetration_fraction=0.,
            jerk_p95_m_s3=500., tracking_mean_m=0.05, tracking_p95_m=0.1)
checks = {}

trust = agreement(base, deepcopy(base), thresholds)
checks['consistent_bias_agrees_but_is_not_accuracy'] = trust['position'] and trust['heading'] and base['endpoint_error_m'] > thresholds['goal_position_m']
bad = deepcopy(base); bad['completed'] = False
checks['terminated_repeat_cannot_license_feedback'] = not agreement(base, bad, thresholds)['position']
bad = deepcopy(base); bad['endpoint_error_vector'] = [float('nan'), 0., 0.]
checks['nonfinite_error_cannot_license_feedback'] = not agreement(base, bad, thresholds)['position']
bad = deepcopy(base); bad['endpoint_error_vector'][0] += 0.02
checks['disagreeing_position_does_not_block_agreeing_heading'] = not agreement(base, bad, thresholds)['position'] and agreement(base, bad, thresholds)['heading']

improved = deepcopy(base); improved['endpoint_error_m'] = 0.1
worse = deepcopy(improved); worse['surface_slip_proxy_fraction'] = 0.4
checks['better_position_cannot_hide_worse_physical_proxy'] = not acceptable(base, worse)[0]
checks['complete_nonregressing_improvement_accepted'] = acceptable(base, improved)[0]
missing = deepcopy(improved); missing['jerk_p95_m_s3'] = None
checks['missing_quality_cannot_be_accepted'] = not acceptable(base, missing)[0]
checks['ties_do_not_replace_incumbent'] = not acceptable(base, deepcopy(base))[0]

choices, decisions = choose_guarded([[base], [deepcopy(base)], [worse]])
checks['rejected_candidates_preserve_input_denominator'] = choices == [0] and len(decisions) == 1 and len(decisions[0]) == 2
incomplete = deepcopy(base); incomplete['completed'] = False
checks['completed_repair_can_replace_incomplete_construction'] = acceptable(incomplete, improved)[0]

metadata = dict(cases=[dict(case_id='synthetic:0')], context_steps=60,
                original_goal=[[0., 0., 1.]], proposed_goal=[[0., 0., 1.]],
                original_heading=[0.], proposed_heading=[0.],
                original_goal_sha256='synthetic_numeric_fixture_not_dataset')
prior = deepcopy(metadata)
result = corrected_conditions(metadata, [base], [deepcopy(base)], config)
shift = np.asarray(result['proposed_goal'])-np.asarray(metadata['proposed_goal'])
checks['correction_bounded_and_original_goal_immutable'] = bool(np.max(np.linalg.norm(shift, axis=-1)) <= config['position_step_m']+1e-8 and metadata == prior and abs(result['proposed_heading'][0]) <= np.deg2rad(config['heading_step_deg'])+1e-12)
muted = corrected_conditions(metadata, [base], [bad], config)
checks['disagreement_zeroes_only_position_update'] = np.array_equal(muted['proposed_goal'], metadata['proposed_goal']) and muted['proposed_heading'][0] != 0

out = ROOT/'results/condition_feedback_guard_v29_20261012'
out.mkdir(exist_ok=True)
record = dict(status='PASS' if all(checks.values()) else 'FAIL',
              checks=checks, evaluation_type='synthetic_numeric_fixture',
              simulation_efficacy_tested=False, innovation_confirmed=False,
              source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                  'scripts/condition_feedback_guard_core_v29_20261012.py',
                  'scripts/check_condition_feedback_guard_v29_20261012.py',
                  'docs/condition_feedback_guard_protocol_v29_20261012.md',
                  'configs/physics_guided_20261011.json')})
(out/'core_numeric_check.json').write_text(json.dumps(record, indent=2), encoding='utf8')
print(json.dumps(record, indent=2))
assert all(checks.values())

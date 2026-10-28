"""New isolated seed/payload/window checks; never import a numerical runtime."""
import copy
import json
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from ldm100k_generation_protocol import pair_seed, pair_output_upper_bytes, validate_diagnostic_scope, SEEDS, MODES

x = dict(segment_id='1' * 64, history_start_feature=0, history_end_feature_exclusive=2,
         start_feature=2, end_feature_exclusive=202, reference_frames=204,
         future_frames=200, primitive_calls=25, final_generated_tail_frames_not_scored=0,
         text='fixture only')
s = dict(summary=dict(diagnostic_family_limit=32, selected_validation_families=32,
    seeds=SEEDS, modes=MODES, diffusion_steps=5, cfg=5,
    transition_metrics_included=False, train_families_used_for_eval=False,
    source_and_mirror_family_disjoint=True, actual_generation_runs=0,
    actual_model_loads=0, segment_count=1, planned_trial_count=10), segments=[x])
p = validate_diagnostic_scope(s)
assert p['planned_pairs'] == 5 and p['planned_trials'] == 10
assert p['output_upper_bound_bytes'] == 5 * pair_output_upper_bytes(200)
assert pair_seed('1' * 64, SEEDS[0]) == pair_seed('1' * 64, SEEDS[0])
assert len({pair_seed('1' * 64, k) for k in SEEDS}) == len(SEEDS)
assert pair_output_upper_bytes(1) < pair_output_upper_bytes(8) < pair_output_upper_bytes(9)
count = 0


def reject(call):
    global count
    try:
        call()
    except (AssertionError, KeyError, TypeError):
        count += 1
        return
    raise AssertionError('Corrupt isolated generation metadata accepted')


for bad in [0, 201, False, 1.5]:
    reject(lambda: pair_output_upper_bytes(bad))
for bad in [0, True, 20261102]:
    reject(lambda: pair_seed('1' * 64, bad))
reject(lambda: pair_seed('x' * 64, SEEDS[0]))
for key, value in [('cfg', 4), ('diffusion_steps', 1), ('seeds', [SEEDS[0]]),
                   ('diagnostic_family_limit', None), ('modes', ['conditional_cfg5']),
                   ('planned_trial_count', 9), ('transition_metrics_included', True),
                   ('train_families_used_for_eval', True)]:
    bad = copy.deepcopy(s)
    bad['summary'][key] = value
    reject(lambda: validate_diagnostic_scope(bad))
for key, value in [('history_start_feature', 1), ('history_end_feature_exclusive', 3),
                   ('end_feature_exclusive', 204), ('primitive_calls', 24),
                   ('final_generated_tail_frames_not_scored', 7), ('text', '')]:
    bad = copy.deepcopy(s)
    bad['segments'][0][key] = value
    reject(lambda: validate_diagnostic_scope(bad))
print(json.dumps(dict(status='LDM100K_GENERATION_NEW_ISOLATED_PROTOCOL_PASSED',
    negative_cases=count, actual_arrays_read=0, models_loaded=0, gpu_jobs=0,
    actual_pair_generation=0, actual_sampler_or_model_acceptance=False)))

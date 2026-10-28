"""New isolated scope boundary/leakage tests, no original arrays/text or models."""
import argparse
import copy
import json
import pathlib
import sys

B = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B / 'scripts'))
from ldm100k_heldout_protocol import frame_bounds, make_segment_plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = pathlib.Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    row = dict(canonical_source='fixture_validation', split='val', mirrored=False,
               physical_filter_passed=True, dof=29, features=69, fps=50, frames50=504,
               output_sha256='a' * 64, annotations=[dict(start_s=0, end_s=10.08, text='fixture motion'),
                   dict(start_s=1, end_s=2, text='fixture overlapping motion'),
                   dict(start_s=0, end_s=.02, text='fixture too short')])
    plan = make_segment_plan([row], {'fixture_train'})
    segments = plan['segments']
    first = [x for x in segments if x['annotation_index'] == 0]
    assert [(x['start_feature'], x['end_feature_exclusive']) for x in first] == [(2, 202), (202, 402), (402, 503)]
    assert sum(x['future_frames'] for x in first) == 501
    assert len(plan['omitted_annotations']) == 1 and not plan['summary']['transition_metrics_included']
    for x in segments:
        assert x['history_end_feature_exclusive'] == x['start_feature']
        assert x['history_end_feature_exclusive'] - x['history_start_feature'] == 2
        assert 1 <= x['future_frames'] <= 200
        assert 0 <= x['final_generated_tail_frames_not_scored'] < 8
    assert frame_bounds(.1001, .2001, 50, 503) == (5, 11, 0)
    assert frame_bounds(0, .06, 50, 503) == (2, 3, 2)
    assert make_segment_plan([row], {'fixture_train'}) == plan
    other = copy.deepcopy(row)
    other['canonical_source'] = 'fixture_validation_other'
    forward = make_segment_plan([row, other], {'fixture_train'}, 1)
    backward = make_segment_plan([other, row], {'fixture_train'}, 1)
    assert forward == backward
    rejected = 0
    cases = []
    for key, value in [('split', 'train'), ('mirrored', True), ('physical_filter_passed', False),
                       ('dof', 23), ('features', 57), ('fps', 60), ('frames50', 1),
                       ('output_sha256', 'z' * 64)]:
        bad = copy.deepcopy(row)
        bad[key] = value
        cases.append(lambda bad=bad: make_segment_plan([bad], {'fixture_train'}))
    cases.extend([lambda: make_segment_plan([row], {'fixture_validation'}),
                  lambda: make_segment_plan([row, row], {'fixture_train'}),
                  lambda: make_segment_plan([row], {'fixture_train'}, 0),
                  lambda: make_segment_plan([row], {'fixture_train'}, 2),
                  lambda: frame_bounds(1, 1, 50, 503),
                  lambda: frame_bounds(-1, 1, 50, 503),
                  lambda: frame_bounds(0, 20, 50, 503),
                  lambda: frame_bounds(0, float('nan'), 50, 503)])
    for call in cases:
        try:
            call()
        except (AssertionError, TypeError):
            rejected += 1
        else:
            raise AssertionError('Invalid scope fixture accepted')
    z = dict(status='LDM100K_HELDOUT_SCOPE_ISOLATED_FIXTURES_PASSED', rejected_cases=rejected,
             chunk_boundary_and_T_minus_1_verified=True, no_future_features_in_initial_history=True,
             overlap_preserved_without_transition_claim=True, deterministic_subset_and_input_order=True,
             actual_original_arrays_read=0, actual_original_text_read=0, actual_models_loaded=0,
             actual_gpu_contexts=0, actual_generation_runs=0, actual_semantic_metrics=0)
    (out / 'SUMMARY.json').write_text(json.dumps(z, indent=2), encoding='utf-8')
    print(json.dumps(z))


if __name__ == '__main__':
    main()

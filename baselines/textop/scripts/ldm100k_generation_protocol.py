"""Pure paired-seed/output budgets for the declared32-family diagnostic."""
import hashlib

SEEDS = [20261028, 20261029, 20261030, 20261031, 20261101]
MODES = ['conditional_cfg5', 'unconditional']


def pair_seed(segment_id, seed):
    assert isinstance(segment_id, str) and len(segment_id) == 64
    assert all(c in '0123456789abcdef' for c in segment_id)
    assert type(seed) is int and seed in SEEDS
    text = 'textop-ldm100k-paired-ddpm-v1:' + segment_id + ':' + str(seed)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') % (2**63 - 1)


def pair_output_upper_bytes(future_frames):
    assert type(future_frames) is int and 0 < future_frames <= 200
    full = ((future_frames + 7) // 8) * 8
    # Each uncompressed NPZ has generated69D, reference69D, qpos36D,
    # contact2D and initial2x69 history, all float32. Headers plus a private
    # per-pair trace/receipt allowance are conservative, not compressed sizes.
    one = (full * 69 + future_frames * (69 + 36 + 2) + 2 * 69) * 4 + 4096
    return 2 * one + 64 * 1024


def validate_diagnostic_scope(scope):
    s = scope['summary']
    assert s['diagnostic_family_limit'] == s['selected_validation_families'] == 32
    assert s['seeds'] == SEEDS and s['modes'] == MODES
    assert s['diffusion_steps'] == s['cfg'] == 5
    assert s['transition_metrics_included'] is False and s['train_families_used_for_eval'] is False
    assert s['source_and_mirror_family_disjoint'] is True
    assert s['actual_generation_runs'] == s['actual_model_loads'] == 0
    segments = scope['segments']
    assert segments and s['segment_count'] == len(segments)
    assert len({x['segment_id'] for x in segments}) == len(segments)
    for x in segments:
        assert x['history_start_feature'] == x['start_feature'] - 2 >= 0
        assert x['history_end_feature_exclusive'] == x['start_feature']
        assert x['end_feature_exclusive'] <= x['reference_frames'] - 1
        assert x['future_frames'] == x['end_feature_exclusive'] - x['start_feature']
        pair_output_upper_bytes(x['future_frames'])
        assert x['primitive_calls'] == (x['future_frames'] + 7) // 8
        assert x['final_generated_tail_frames_not_scored'] == x['primitive_calls'] * 8 - x['future_frames']
        assert isinstance(x['text'], str) and x['text'].strip()
    assert s['planned_trial_count'] == len(segments) * len(SEEDS) * len(MODES)
    return dict(planned_pairs=len(segments) * len(SEEDS), planned_trials=s['planned_trial_count'],
                output_upper_bound_bytes=sum(pair_output_upper_bytes(x['future_frames']) for x in segments) * len(SEEDS),
                actual_completed_pairs=0, actual_completed_trials=0)

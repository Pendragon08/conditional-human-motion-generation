"""Deterministic PRIVATE heldout segment scope; no arrays, models or acceptance.

Paper limits segment length to200 frames. This reproduction explicitly fixes
timestamp rounding, two-frame past context and diagnostic subset selection.
They are declared protocol choices, not an undocumented exact-paper claim.
"""
import decimal
import hashlib
import json

SEEDS = [20261028, 20261029, 20261030, 20261031, 20261101]
MODES = ['conditional_cfg5', 'unconditional']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def frame_bounds(start_s, end_s, fps, feature_frames):
    start, end = decimal.Decimal(str(start_s)), decimal.Decimal(str(end_s))
    assert start.is_finite() and end.is_finite() and 0 <= start < end
    assert type(fps) is int and fps == 50
    assert type(feature_frames) is int and feature_frames >= 1
    assert end <= decimal.Decimal(feature_frames + 1) / fps
    first = int((start * fps).to_integral_value(rounding=decimal.ROUND_FLOOR))
    last = int((end * fps).to_integral_value(rounding=decimal.ROUND_CEILING))
    return max(2, first), min(feature_frames, last), max(0, 2 - first)


def make_segment_plan(rows, train_families, maximum_families=None):
    """Rows contain only actual accepted unmirrored validation metadata.

    All annotation windows are retained separately, including overlaps and
    duplicate text; they are not stitched into invented transition targets.
    Full arrays/CLIP and the model remain untouched until separately launched.
    """
    assert isinstance(rows, list) and rows
    train_families = set(train_families)
    sources = [r['canonical_source'] for r in rows]
    assert len(sources) == len(set(sources)) and not set(sources) & train_families
    for row in rows:
        assert row['split'] == 'val' and row['mirrored'] is False
        assert row['physical_filter_passed'] is True
        assert row['dof'] == 29 and row['features'] == 69 and row['fps'] == 50
        assert type(row['frames50']) is int and row['frames50'] >= 2
        assert len(row['output_sha256']) == 64
        assert all(c in '0123456789abcdef' for c in row['output_sha256'])
    ordered = sorted(rows, key=lambda r: hashlib.sha256(r['canonical_source'].encode()).hexdigest())
    if maximum_families is not None:
        assert type(maximum_families) is int and 0 < maximum_families <= len(ordered)
        ordered = ordered[:maximum_families]
    segments, omissions = [], []
    for row in ordered:
        feature_frames = row['frames50'] - 1  # pairwise69D encoder has T-1 rows
        for annotation_index, ann in enumerate(row['annotations']):
            assert isinstance(ann['text'], str) and ann['text'].strip()
            first, last, prefix_omitted = frame_bounds(ann['start_s'], ann['end_s'], 50, feature_frames)
            if first >= last:
                omissions.append(dict(canonical_source=row['canonical_source'], annotation_index=annotation_index,
                                      reason='no_future_after_two_past_features_and_T_minus_1_boundary',
                                      physical_failure_claimed=False))
                continue
            for start in range(first, last, 200):
                end = min(start + 200, last)
                key = dict(canonical_source=row['canonical_source'], annotation_index=annotation_index,
                           start_feature=start, end_feature_exclusive=end,
                           text_sha256=hashlib.sha256(ann['text'].encode()).hexdigest())
                segments.append(dict(key, segment_id=digest(key), text=ann['text'],
                                     history_start_feature=start - 2, history_end_feature_exclusive=start,
                                     reference_frames=row['frames50'], reference_output_sha256=row['output_sha256'],
                                     future_frames=end - start, primitive_calls=(end - start + 7) // 8,
                                     final_generated_tail_frames_not_scored=((end - start + 7) // 8) * 8 - (end - start),
                                     annotation_prefix_features_omitted=prefix_omitted if start == first else 0))
    assert segments and len(segments) == len({x['segment_id'] for x in segments})
    summary = dict(schema='textop.ldm100k_heldout_scope.v1', fps=50, feature_dim=69, dof=29,
                   source_validation_families=len(rows), selected_validation_families=len(ordered),
                   diagnostic_family_limit=maximum_families, maximum_segment_future_frames=200,
                   segment_count=len(segments), omitted_annotation_windows=len(omissions),
                   scored_future_frames=sum(x['future_frames'] for x in segments),
                   primitive_calls_per_seed_and_mode=sum(x['primitive_calls'] for x in segments),
                   seeds=SEEDS, modes=MODES, diffusion_steps=5, cfg=5,
                   pairing='same_seed_and_two_past_reference_features; later_history_generated_only',
                   timestamp_rounding='floor(start_s*50),ceil(end_s*50); target_start>=2; target_end<=T-1',
                   transition_metrics_included=False,
                   overlapping_annotations='separate_segment_tasks_preserved; no_invented_transition_order',
                   source_and_mirror_family_disjoint=True, train_families_used_for_eval=False,
                   text_motion_feature_extractor_accepted=False, semantic_metrics_computed=False,
                   actual_generation_runs=0, actual_model_loads=0,
                   source_split_is_original_babel_validation=True)
    summary['planned_trial_count'] = len(segments) * len(SEEDS) * len(MODES)
    summary['private_segment_plan_sha256_canonical'] = digest(segments)
    return dict(summary=summary, segments=segments, omitted_annotations=omissions)

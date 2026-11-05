"""Explicit, opt-in views of discrete contact labels; no training integration.

Bins partition source *frame indices*. The caller supplies bin edges and
validity, so the source frequency and each label's temporal meaning are not
inferred. A bin's `any_known_positive` is a three-valued logical reduction,
not proof of contact throughout the interval or of physical force.
"""
from __future__ import annotations

import numpy as np


def contact_interval_view(labels, valid, edges, *, source_frame_start=0,
                          declared_source_fps=None):
    labels = np.asarray(labels)
    valid = np.asarray(valid)
    edges = np.asarray(edges)
    if labels.dtype != np.bool_ or valid.dtype != np.bool_:
        raise ValueError('labels and validity must be boolean')
    if labels.ndim != 2 or valid.shape != labels.shape or not labels.shape[0] or not labels.shape[1]:
        raise ValueError('labels and validity must have the same nonempty (frames, channels) shape')
    if edges.ndim != 1 or edges.dtype.kind not in 'iu' or len(edges) < 2:
        raise ValueError('bin edges must be a one-dimensional integer array')
    if edges[0] != 0 or edges[-1] != len(labels) or np.any(edges[1:] <= edges[:-1]):
        raise ValueError('strict bin edges must cover every input frame exactly once')
    if isinstance(source_frame_start, (bool, np.bool_)) or not isinstance(source_frame_start, (int, np.integer)) or source_frame_start < 0:
        raise ValueError('source frame start must be a nonnegative integer')
    if int(source_frame_start) + len(labels) > np.iinfo(np.int64).max:
        raise ValueError('absolute source frame indices exceed int64 range')
    if declared_source_fps is not None:
        if isinstance(declared_source_fps, (bool, np.bool_)) or not np.isscalar(declared_source_fps):
            raise ValueError('declared rate must be a positive finite scalar or None')
        try:
            declared_source_fps = float(declared_source_fps)
        except (TypeError, ValueError) as exc:
            raise ValueError('declared rate must be numeric') from exc
        if not np.isfinite(declared_source_fps) or declared_source_fps <= 0:
            raise ValueError('declared rate must be positive and finite')
    any_positive, known, fully_observed, lower, upper = [], [], [], [], []
    for first, last in zip(edges[:-1], edges[1:]):
        present = labels[first:last] & valid[first:last]
        observed = valid[first:last]
        positive = present.any(axis=0)
        all_observed = observed.all(axis=0)
        any_positive.append(positive)
        known.append(positive | all_observed)
        fully_observed.append(all_observed)
        denominator = last - first
        lower.append(present.sum(axis=0) / denominator)
        upper.append((present.sum(axis=0) + (~observed).sum(axis=0)) / denominator)
    begin = edges[:-1].astype(np.int64) + int(source_frame_start)
    end = edges[1:].astype(np.int64) + int(source_frame_start)
    return {'source_frame_begin_inclusive': begin,
            'source_frame_end_exclusive': end,
            'first_sample_positive': (labels[edges[:-1]] & valid[edges[:-1]]).copy(),
            'first_sample_known': valid[edges[:-1]].copy(),
            'any_known_positive': np.stack(any_positive),
            'any_known_positive_known': np.stack(known),
            'fully_observed': np.stack(fully_observed),
            'positive_frame_fraction_lower': np.stack(lower),
            'positive_frame_fraction_upper': np.stack(upper),
            'declared_source_fps': declared_source_fps}

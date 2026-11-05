"""Preserve HumanML reference frame times when preparing a true 20 Hz grid.

The source-specific reference FPS and measured frame identity must be verified
before using these interpolation weights on a different AMASS export. This
module computes times only; callers must use proper rotation interpolation.
"""
from bisect import bisect_right
import math


def aligned_grid(frame_count, reference_fps, start, stop, *, prefix_trim=0, output_fps=20.0):
    if frame_count < 2 or not math.isfinite(reference_fps) or reference_fps < 20:
        raise ValueError('invalid source dimensions or fps')
    if not math.isfinite(output_fps) or output_fps <= 0 or prefix_trim < 0:
        raise ValueError('invalid output fps or trim')
    stride = int(reference_fps / 20)
    sampled = list(range(0, frame_count, stride))[prefix_trim:]
    if start < 0 or stop < -1 or start >= len(sampled) or (stop != -1 and stop > len(sampled)):
        raise ValueError('source clip out of bounds')
    indices = sampled[start:stop]
    if len(indices) < 2:
        raise ValueError('source clip too short')
    # Each 20-Hz point is measured from the exact first sampled recording frame.
    duration = (indices[-1] - indices[0]) / reference_fps
    count = int(math.floor(duration * output_fps + 1e-10)) + 1
    source_times = [(i - indices[0]) / reference_fps for i in indices]
    weights = []
    for j in range(count):
        t = j / output_fps
        right = bisect_right(source_times, t)
        if right == 0:
            left = right = 0
        elif right == len(indices):
            left = right = len(indices) - 1
        else:
            left = right - 1
        alpha = 0.0 if left == right else (t - source_times[left]) / (source_times[right] - source_times[left])
        weights.append({'time':t,'left_frame':indices[left],'right_frame':indices[right],'alpha':alpha})
    return {'reference_stride':stride,'actual_reference_sample_fps':reference_fps/stride,
            'source_indices':indices,'source_duration_s':duration,'output_fps':output_fps,'weights':weights}

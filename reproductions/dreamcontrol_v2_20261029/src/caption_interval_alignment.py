"""Conservative time association of HumanML captions and feature intervals.

0/0 is the legacy whole-clip convention, never a literal zero-second label.
Timed tags are interpreted as seconds only when the caller explicitly selects
that convention. Time association does not validate words or source identity.
"""
import math
import numpy as np


def associate_caption(*, clip_sample_times, feature_sample_times, start_tag, end_tag,
                      timed_tag_convention, tolerance=1e-9):
    clip=np.asarray(clip_sample_times,dtype=np.float64)
    feature=np.asarray(feature_sample_times,dtype=np.float64)
    for name, value in (('clip',clip),('feature',feature)):
        if value.ndim!=1 or len(value)<2 or not np.isfinite(value).all() or not np.all(np.diff(value)>0):
            raise ValueError(name+' times must be finite and strictly increasing')
    if not math.isfinite(tolerance) or tolerance<0:raise ValueError('invalid tolerance')
    if feature[0]<clip[0]-tolerance or feature[-1]>clip[-1]+tolerance:
        raise ValueError('feature sample times extend beyond known clip samples')
    try:a=float(start_tag);b=float(end_tag)
    except (TypeError,ValueError):raise ValueError('invalid time tags')
    if not math.isfinite(a) or not math.isfinite(b):raise ValueError('nonfinite tags are unknown, not zero')
    if timed_tag_convention not in ('seconds_relative_to_clip','unknown'):
        raise ValueError('unknown timed tag convention')
    if a==0 and b==0:
        label='whole_clip_convention';lo=float(clip[0]);hi=float(clip[-1])
    else:
        if not 0<=a<b:raise ValueError('invalid caption interval')
        if timed_tag_convention!='seconds_relative_to_clip':
            return {'status':'unknown_time_convention','matching_feature_intervals':[],
                    'caption_accepted':False,'reason':'timed tags need a verified time convention'}
        label='seconds_relative_to_clip';lo=float(clip[0]+a);hi=float(clip[0]+b)
        if lo<clip[0]-tolerance or hi>clip[-1]+tolerance:
            return {'status':'out_of_known_sample_range','matching_feature_intervals':[],
                    'caption_accepted':False,'reason':'tag extends beyond known clip samples'}
    mask=(feature[:-1]>=lo-tolerance)&(feature[1:]<=hi+tolerance)
    selected=np.flatnonzero(mask)
    feature_covers_caption=(abs(feature[0]-lo)<=tolerance and abs(feature[-1]-hi)<=tolerance)
    return {'status':'associated' if len(selected) else 'no_fully_covered_intervals',
            'tag_interpretation':label,'annotation_start_time':lo,'annotation_end_time':hi,
            'matching_feature_intervals':selected.tolist(),
            'feature_covers_full_annotation_sample_extent':bool(feature_covers_caption),
            'caption_accepted':False,
            'reason':'time association does not establish caption semantics or source identity'}

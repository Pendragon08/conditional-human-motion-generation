"""Private metadata-only dance candidate preparation; not a simulation launcher.

Inputs must be independently authenticated before use. No arrays, models, SDK,
physics, motion reconstruction or actual rollout is performed here.
"""
import hashlib
import re

DANCE = re.compile(r"\b(dance|dances|dancing|dancer|ballet|salsa|tango|waltz|ballroom|hip[ -]?hop|breakdanc(?:e|ing))\b", re.I)


def candidates(validation_segments, validation_sources, train_families, minimum_frames=500, maximum_frames=1500):
    """Select every qualified contiguous annotation, before seeing physical outcomes.

    Frames refer to 50Hz feature intervals. Adjacent pieces are metadata portions
    of the same single annotated original motion, never concatenated generation.
    Caller still must verify all source hashes, exact arrays and numeric reference.
    """
    if type(minimum_frames) is not int or type(maximum_frames) is not int or not 1 <= minimum_frames <= maximum_frames <= 1999:
        raise ValueError('invalid duration bounds')
    if not isinstance(validation_sources, list) or not validation_sources or not isinstance(validation_segments, list):
        raise ValueError('invalid source scope')
    train = set(train_families)
    index = {}
    for row in validation_sources:
        k = row.get('canonical_source')
        if not isinstance(k, str) or not k or k in train or k in index or row.get('split') != 'val' or row.get('mirrored') is not False:
            raise ValueError('source identity, split or leakage')
        if type(row.get('frames50')) is not int or row['frames50'] < 2:
            raise ValueError('invalid source duration')
        h = row.get('output_sha256')
        if not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
            raise ValueError('invalid source hash')
        index[k] = row
    groups = {}
    for x in validation_segments:
        name=x.get('canonical_source'); text=x.get('text'); h=x.get('text_sha256')
        if name not in index or not isinstance(text,str) or not text or h != hashlib.sha256(text.encode()).hexdigest():
            raise ValueError('segment identity or text binding')
        if x.get('reference_output_sha256') != index[name]['output_sha256']:
            raise ValueError('reference hash mismatch')
        start=x.get('start_feature'); end=x.get('end_feature_exclusive'); ai=x.get('annotation_index')
        if type(start) is not int or type(end) is not int or type(ai) is not int or ai<0 or not 0<=start<end<=index[name]['frames50']-1:
            raise ValueError('invalid interval')
        if not DANCE.search(text):
            continue
        key=(name,ai,h)
        groups.setdefault(key,[]).append((start,end))
    result=[]
    for (name,ai,h),intervals in groups.items():
        intervals.sort()
        if any(a[1]!=b[0] for a,b in zip(intervals,intervals[1:])):
            raise ValueError('annotation parts overlap, duplicate or have gaps')
        first=intervals[0][0]; last=intervals[-1][1]
        if last-first<minimum_frames:
            continue
        selected_last=min(last,first+maximum_frames)
        result.append(dict(canonical_source=name,annotation_index=ai,text_sha256=h,
            source_file=index[name]['output_file'],source_sha256=index[name]['output_sha256'],
            first_feature=first,end_feature_exclusive=selected_last,reference_fps=50,
            interval_count=selected_last-first, annotation_end_feature_exclusive=last,
            selection='all matching validation annotations; deterministic first bounded window; no physical outcome filtering'))
    result.sort(key=lambda r:(hashlib.sha256(r['canonical_source'].encode()).hexdigest(),r['annotation_index'],r['text_sha256']))
    return result

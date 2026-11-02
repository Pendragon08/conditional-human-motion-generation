"""Metadata-only bound for future private VAL dance FK, not an executor.

One source annotation yields one capped continuous interval; no motion arrays,
SDK, model, physical tracking or media output is read or produced here.
"""
import hashlib
from pathlib import PurePosixPath


def plan(candidates, train_families):
    if not isinstance(candidates,list) or not candidates:
        raise ValueError('nonempty validated dance candidates required')
    train=set(train_families); out=[]; seen=set(); output_bytes=0
    for x in candidates:
        family=x.get('canonical_source'); idx=x.get('annotation_index'); h=x.get('source_sha256'); th=x.get('text_sha256')
        if not isinstance(family,str) or not family or family in train or type(idx) is not int or idx<0:
            raise ValueError('leakage or invalid identity')
        if any(not isinstance(y,str) or len(y)!=64 or any(c not in '0123456789abcdef' for c in y) for y in (h,th)):
            raise ValueError('invalid fingerprint')
        source=x.get('source_file')
        if not isinstance(source,str) or not PurePosixPath(source).is_absolute() or '..' in PurePosixPath(source).parts:
            raise ValueError('invalid private source path')
        first=x.get('first_feature'); last=x.get('end_feature_exclusive'); end=x.get('annotation_end_feature_exclusive'); n=x.get('interval_count')
        if any(type(v) is not int for v in (first,last,end,n)) or first<0 or not (first<last<=end) or n!=last-first or not 500<=n<=1500 or x.get('reference_fps')!=50:
            raise ValueError('invalid contiguous reference bounds')
        key=(family,idx,th)
        if key in seen:
            raise ValueError('duplicate annotation')
        seen.add(key)
        pose_count=n+1
        bound=pose_count*2152+8192
        output_bytes+=bound
        out.append(dict(source_family=family,annotation_index=idx,text_sha256=th,source_file=source,source_sha256=h,
            first_pose_index=first,last_pose_index_inclusive=last,control_intervals=n,pose_count=pose_count,
            maximum_npz_bytes=bound,reference_fps=50,tracking_performed=False))
    return dict(candidates=out,candidate_count=len(out),target_pose_count=sum(x['pose_count'] for x in out),
        npz_upper_bytes=output_bytes,asset_allowance_bytes=128*1024**2,log_allowance_bytes=128*1024**2,
        reserve_bytes=2*1024**3,required_free_bytes=output_bytes+128*1024**2+128*1024**2+2*1024**3,
        note='metadata-only estimate, not a disk reservation or actual output; owner, source bytes and array contents require independent verification')

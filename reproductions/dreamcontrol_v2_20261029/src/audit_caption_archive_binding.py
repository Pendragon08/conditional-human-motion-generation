"""Re-read original HumanML captions and check nominal sample-time pairing.

This is a technical binding audit: raw 0/0 means whole indexed clip under the
HumanML convention, not proof of the source clock, semantics, or data license.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def annotation_range(tags, clip_range):
    if not isinstance(tags,list) or len(tags)!=2:raise ValueError('two raw caption tags required')
    try:start,end=(float(x) for x in tags)
    except (ValueError,TypeError):raise ValueError('invalid caption tags') from None
    if not math.isfinite(start) or not math.isfinite(end):raise ValueError('non-finite caption time')
    if start==0 and end==0:return tuple(clip_range),'whole_clip_convention'
    if start<0 or end<=start:raise ValueError('invalid annotated interval')
    if start<clip_range[0] or end>clip_range[1]:raise ValueError('annotation exceeds declared clip interval')
    return (start,end),'timed_segment'

def audit_caption_archive(records, *, archive=None, evidence_root=None, evidence_manifest=None,
                          state_root=None, state_manifest=None):
    accepted=[r for r in records if r.get('caption_accepted') is True]
    config=(archive,evidence_root,evidence_manifest,state_root,state_manifest)
    if all(x is None for x in config):return {'status':'missing_archive_binding','training_ready':False}
    if any(x is None for x in config):raise ValueError('caption archive binding requires all five paths')
    archive=Path(archive).resolve(strict=True);root=Path(evidence_root).resolve(strict=True)
    bundle_path=Path(evidence_manifest).resolve(strict=True);state_root=Path(state_root).resolve(strict=True)
    state_manifest=Path(state_manifest).resolve(strict=True)
    archive_hash=digest(archive);bundle_hash=digest(bundle_path);state_manifest_hash=digest(state_manifest)
    evidence=json.loads(bundle_path.read_text())['receipts'];states=json.loads(state_manifest.read_text())['records']
    by_id={x['sample_id']:x for x in evidence};by_state={x['sample_id']:x for x in states}
    ids={r['sample_id'] for r in accepted}
    if len(by_id)!=len(evidence) or len(by_state)!=len(states) or set(by_id)!=ids or not ids.issubset(by_state):
        raise ValueError('caption receipt or state coverage mismatch')
    results=[]
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        for row in accepted:
            sid=row['sample_id'];entry=by_id[sid];relative=Path(entry['receipt_path'])
            receipt=(root/relative).resolve(strict=True)
            if relative.is_absolute() or not receipt.is_relative_to(root) or digest(receipt)!=entry['receipt_sha256']:
                raise ValueError('caption receipt path or hash mismatch')
            r=json.loads(receipt.read_text());state=by_state[sid]
            if r['sample_id']!=sid or r['caption']!=row['caption'] or r['candidate_sha256']!=row['feature_sha256'] or state['feature_sha256']!=row['feature_sha256']:
                raise ValueError('caption, feature or state binding mismatch')
            if r['caption_archive_sha256']!=archive_hash:raise ValueError('caption archive hash mismatch')
            m=re.search(r'(?:^|\s)(texts/[0-9]{6}\.txt)(?:;|$)',row['caption_source'])
            if not m:raise ValueError('caption member path not explicit')
            member=m.group(1)
            if names.count(member)!=1:raise ValueError('missing or duplicate caption member')
            info=z.getinfo(member)
            if info.file_size>65536:raise ValueError('caption member too large')
            raw=z.read(info)
            if hashlib.sha256(raw).hexdigest()!=r['caption_member_sha256']:raise ValueError('caption member hash mismatch')
            lines=raw.decode('utf8').splitlines();idx=r['selected_line']
            if isinstance(idx,bool) or not isinstance(idx,int) or idx<0 or idx>=len(lines):raise ValueError('invalid selected line')
            fields=lines[idx].split('#')
            if len(fields)!=4 or fields[0]!=row['caption'] or fields[2:]!=r['tags']:raise ValueError('selected caption or raw tags mismatch')
            clip=r['source_range_s'];reviewed=r['sample_range_s']
            if len(clip)!=2 or len(reviewed)!=2 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in clip+reviewed) or clip[0]>clip[1]:
                raise ValueError('invalid clip or reviewed interval')
            annotation,mode=annotation_range(r['tags'],clip)
            path=(state_root/state['state_path']).resolve(strict=True)
            if not path.is_relative_to(state_root) or digest(path)!=state['state_sha256']:raise ValueError('state file hash mismatch')
            with np.load(path,allow_pickle=False) as f:t=np.asarray(f['sample_times'],dtype=np.float64)
            if t.ndim!=1 or len(t)<3 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('invalid state sample times')
            if abs(float(t[0])-reviewed[0])>1e-7 or abs(float(t[-1])-reviewed[1])>1e-7:
                raise ValueError('sample timeline does not match reviewed interval')
            if float(t[0])<annotation[0]-1e-7 or float(t[-1])>annotation[1]+1e-7:
                raise ValueError('feature-supporting samples exceed caption interval')
            if digest(path)!=state['state_sha256'] or digest(receipt)!=entry['receipt_sha256']:raise RuntimeError('caption input changed')
            results.append({'sample_id':sid,'archive_member':member,'selected_line_zero_based':idx,'raw_tags':r['tags'],
                            'tag_interpretation':mode,'nominal_sample_range_s':[float(t[0]),float(t[-1])],
                            'declared_annotation_range_s':list(annotation),'feature_intervals':len(t)-1})
    if digest(archive)!=archive_hash or digest(bundle_path)!=bundle_hash or digest(state_manifest)!=state_manifest_hash:
        raise RuntimeError('caption archive inputs changed')
    return {'status':'nominal_archive_and_time_binding_checked','cases':results,'archive_sha256':archive_hash,
            'training_ready':False,'limitations':['Whole-clip and timestamp interpretations follow nominal HumanML conventions.',
            'Source-range and sample times are not independently measured clocks.',
            'This validates original text binding, not true motion semantics, source identity, license or physics.']}

"""Validate provenance and time binding of separately marked visual annotations.

This does not independently judge semantic truth or accept training data.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def resolve(root,name,wanted):
    if not isinstance(name,str) or not name or Path(name).is_absolute() or not isinstance(wanted,str) or re.fullmatch('[0-9a-f]{64}',wanted) is None:
        raise ValueError('invalid evidence path or hash')
    path=(root/name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or digest(path)!=wanted:raise ValueError('evidence path or hash mismatch')
    return path

def audit_manual_captions(records, *, evidence_root=None, evidence_manifest=None, state_root=None, state_manifest=None):
    if not isinstance(records,list):raise ValueError('records must be a list')
    if not records:
        if any(v is not None for v in (evidence_root,evidence_manifest,state_root,state_manifest)):
            raise ValueError('unexpected manual caption evidence without manual captions')
        return {'status':'no_manual_captions','cases':[],'training_ready':False}
    if any(v is None for v in (evidence_root,evidence_manifest,state_root,state_manifest)):
        raise ValueError('manual captions require evidence and state paths')
    root=Path(evidence_root).resolve(strict=True);state_root=Path(state_root).resolve(strict=True)
    ep=Path(evidence_manifest).resolve(strict=True);sp=Path(state_manifest).resolve(strict=True)
    ehash,shash=digest(ep),digest(sp)
    evidence=json.loads(ep.read_text())['receipts'];states=json.loads(sp.read_text())['records']
    eidx={x['sample_id']:x for x in evidence};sidx={x['sample_id']:x for x in states};ids={r['sample_id'] for r in records}
    if len(ids)!=len(records) or len(eidx)!=len(evidence) or len(sidx)!=len(states) or set(eidx)!=ids or not ids.issubset(sidx):
        raise ValueError('manual caption evidence coverage mismatch')
    cases=[]
    for row in records:
        sid=row['sample_id'];entry=eidx[sid];state=sidx[sid]
        if row.get('caption_origin')!='manual_visual' or row.get('caption_accepted') is not True:
            raise ValueError('manual caption must declare origin and review status')
        ap=resolve(root,entry['annotation_path'],entry['annotation_sha256'])
        vp=resolve(root,entry['audit_path'],entry['audit_sha256'])
        annotation=json.loads(ap.read_text());verification=json.loads(vp.read_text())
        if (row.get('caption_review_sha256')!=entry['annotation_sha256'] or verification.get('annotation_sha256')!=entry['annotation_sha256'] or
            annotation.get('sample_id')!=sid or annotation.get('caption')!=row.get('caption') or
            annotation.get('feature_sha256')!=row.get('feature_sha256') or state.get('feature_sha256')!=row.get('feature_sha256') or
            annotation.get('semantic_supported_for_exported_window') is not True or annotation.get('training_ready') is not False or
            verification.get('training_ready') is not False):raise ValueError('manual caption does not bind record')
        if (not isinstance(annotation.get('caption_origin'),str) or
            not annotation['caption_origin'].startswith('manual visual annotation') or
            'not provided by' not in annotation['caption_origin']):
            raise ValueError('manual annotation source not explicit')
        if not isinstance(annotation.get('reviewer'),str) or not annotation['reviewer'].strip():raise ValueError('manual reviewer missing')
        rate=entry.get('declared_source_fps')
        if isinstance(rate,bool) or not isinstance(rate,(int,float)) or not math.isfinite(rate) or rate<=0:raise ValueError('invalid declared source fps')
        path=resolve(state_root,state['state_path'],state['state_sha256'])
        with np.load(path,allow_pickle=False) as f:
            t=np.asarray(f['sample_times'],dtype=np.float64)
            raw_source_frames=np.asarray(f['raw_source_frames']) if 'raw_source_frames' in f else None
        mapping=annotation.get('timeline_mapping','integer_source_frames')
        if mapping=='integer_source_frames':
            frames=np.asarray(annotation['source_frames'])
            if not np.issubdtype(frames.dtype,np.integer):raise ValueError('source frames must be integers')
            bounds=[int(frames[0]),int(frames[-1])] if len(frames) else []
        elif mapping=='linear_source_frame_positions':
            frames=np.asarray(annotation['sample_source_frame_positions'],dtype=np.float64)
            bounds=annotation['source_frame_bounds']
            if (not isinstance(bounds,list) or len(bounds)!=2 or
                any(isinstance(v,bool) or not isinstance(v,int) for v in bounds) or
                bounds[0]<0 or bounds[1]<=bounds[0] or frames.ndim!=1 or not len(frames) or
                not np.isfinite(frames).all() or frames[0]<bounds[0]-1e-7 or frames[-1]>bounds[1]+1e-7):
                raise ValueError('invalid interpolated source frame bounds')
            if (raw_source_frames is None or raw_source_frames.ndim!=1 or len(raw_source_frames)<2 or
                not np.issubdtype(raw_source_frames.dtype,np.integer) or
                np.any(np.diff(raw_source_frames)!=1) or
                [int(raw_source_frames[0]),int(raw_source_frames[-1])]!=bounds):
                raise ValueError('source frame bounds do not match hashed state support')
        else:raise ValueError('unknown manual timeline mapping')
        if (t.ndim!=1 or len(t)<3 or frames.ndim!=1 or len(t)!=len(frames) or
            not np.isfinite(t).all() or np.any(np.diff(frames)<=0) or not np.allclose(t,frames.astype(float)/rate,rtol=0,atol=1e-8) or
            annotation['feature_rows']!=len(t)-1):raise ValueError('manual caption timeline differs from state')
        if not np.allclose(annotation['nominal_time_s'],[t[0],t[-1]],rtol=0,atol=1e-8):raise ValueError('manual annotation endpoints mismatch')
        if digest(path)!=state['state_sha256'] or digest(ap)!=entry['annotation_sha256'] or digest(vp)!=entry['audit_sha256']:
            raise RuntimeError('manual annotation evidence changed')
        cases.append({'sample_id':sid,'annotation_sha256':entry['annotation_sha256'],
                      'source_frame_range':bounds,'timeline_mapping':mapping,
                      'declared_source_fps':float(rate),'feature_intervals':len(t)-1})
    if digest(ep)!=ehash or digest(sp)!=shash:raise RuntimeError('manual evidence manifest changed')
    return {'status':'manual_caption_binding_checked','cases':cases,'training_ready':False,
            'limitations':['A manual review is a subjective visual judgment, not original source text.',
                           'Declared source rate and source identity are not independently established.',
                           'Hashed state frame support does not establish filter context or original-file identity.',
                           'No source/license, physical or overall training acceptance.']}

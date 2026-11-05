"""Recompute SSM source/reference marker correspondence for dataset preflight.

The check is restricted to numeric NPZ arrays. Identical markers show that the
two local files share measured frame values; they do not prove their official
HumanML intermediate identity, real capture rate, model equivalence or license.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _labels(values):
    return [x.decode('utf8') if isinstance(x,bytes) else str(x) for x in values]


def audit_ssm_source(record,alignment,*,receipt_path,receipt_sha256):
    path=Path(receipt_path).resolve(strict=True)
    if digest(path)!=receipt_sha256:raise ValueError('source receipt hash mismatch')
    receipt=json.loads(path.read_text(encoding='utf8'))
    if (alignment.get('motion_id')!=record.get('sample_id') or alignment.get('mirrored') is not False or
        alignment.get('source_path')!=record.get('source_id') or receipt.get('source_group')!=record.get('source_id') or
        Path(alignment.get('source_receipt','')).resolve()!=path):
        raise ValueError('sample and source mapping are inconsistent')
    if receipt.get('timeline',{}).get('status')!='FRAME_IDENTITY_VERIFIED_BY_MEASURED_MARKERS':
        raise ValueError('prior measured marker receipt required')
    source=Path(receipt['smplx_file']).resolve(strict=True);ref=Path(receipt['reference_file']).resolve(strict=True)
    if digest(source)!=receipt['smplx_sha256'] or digest(ref)!=receipt['reference_sha256']:
        raise ValueError('source or reference hash mismatch')
    with np.load(source,allow_pickle=False) as f:
        a=np.asarray(f['markers'],dtype=np.float64);la=_labels(f['labels'])
        source_rate=float(f['mocap_frame_rate']);source_duration=float(f['mocap_time_length'])
    with np.load(ref,allow_pickle=False) as f:
        b=np.asarray(f['marker_data'],dtype=np.float64);lb=_labels(f['marker_labels'])
        ref_rate=float(f['mocap_framerate'])
    if (a.ndim!=3 or b.ndim!=3 or a.shape[-1]!=3 or b.shape[-1]!=3 or a.shape[0]!=b.shape[0] or
        len(la)!=a.shape[1] or len(lb)!=b.shape[1] or len(la)!=len(set(la)) or len(lb)!=len(set(lb)) or
        not np.isfinite(source_rate) or source_rate<=0 or not np.isfinite(ref_rate) or ref_rate<=0 or
        not np.isfinite(source_duration) or source_duration<=0):
        raise ValueError('malformed marker arrays or nominal time metadata')
    common=sorted(set(la)&set(lb))
    if not common:raise ValueError('no common marker labels')
    x=a[:,[la.index(k) for k in common],:];y=b[:,[lb.index(k) for k in common],:]
    valid_x=np.isfinite(x).all(axis=2)&np.any(x!=0,axis=2)
    valid_y=np.isfinite(y).all(axis=2)&np.any(y!=0,axis=2)
    if not np.array_equal(valid_x,valid_y):raise ValueError('marker visibility differs')
    count=valid_x.sum(axis=1)
    if np.any(count<1):raise ValueError('an input frame has no comparable markers')
    delta=np.abs(x[valid_x]-y[valid_x]);maximum=float(delta.max())
    if maximum!=0:raise ValueError('measured markers do not match frame by frame')
    if (digest(path)!=receipt_sha256 or digest(source)!=receipt['smplx_sha256'] or
        digest(ref)!=receipt['reference_sha256']):raise RuntimeError('source evidence changed during audit')
    return {'sample_id':record['sample_id'],'source_sha256':receipt['smplx_sha256'],
            'reference_sha256':receipt['reference_sha256'],'receipt_sha256':receipt_sha256,
            'frames_compared':len(a),'common_marker_labels':len(common),
            'minimum_valid_markers_per_frame':int(count.min()),'valid_coordinate_count':int(valid_x.sum())*3,
            'marker_max_abs_difference_m':maximum,'source_header_rate_hz':source_rate,
            'source_header_duration_s':source_duration,'reference_header_rate_hz':ref_rate,
            'header_rates_disagree':abs(source_rate-ref_rate)>1e-4,
            'official_intermediate_identity_accepted':False,'actual_capture_time_accepted':False,
            'training_ready':False,
            'limitations':['Local marker equality does not establish released HumanML intermediate or body-model equivalence.',
                           'Reference rate is metadata, not an independently measured clock.',
                           'No license, subject independence, contact or dynamics acceptance.']}


def audit_ssm_source_bundle(records,*,alignment_path=None,evidence_manifest=None):
    if not isinstance(records,list) or not records:raise ValueError('nonempty records required')
    if (alignment_path is None)!=(evidence_manifest is None):
        raise ValueError('SSM alignment and evidence manifest must be supplied together')
    if alignment_path is None:
        return {'marker_evidence_cases':0,'samples_without_marker_evidence':[r['sample_id'] for r in records],
                'training_ready':False,'limitations':['Source identity and timing are not accepted.']}
    index=Path(alignment_path).resolve(strict=True);manifest=Path(evidence_manifest).resolve(strict=True)
    index_hash,manifest_hash=digest(index),digest(manifest)
    config=json.loads(manifest.read_text(encoding='utf8'))
    if config.get('alignment_sha256')!=index_hash or not isinstance(config.get('receipts'),list):
        raise ValueError('SSM evidence does not pin its alignment index')
    by_id={}
    for entry in config['receipts']:
        sid=entry.get('sample_id')
        if not isinstance(sid,str) or not sid or sid in by_id:raise ValueError('duplicate or missing SSM evidence ID')
        by_id[sid]=entry
    samples={r.get('sample_id'):r for r in records}
    if len(samples)!=len(records) or not set(by_id).issubset(samples):raise ValueError('SSM evidence IDs do not match dataset')
    alignment={}
    for line in index.read_text(encoding='utf8').splitlines():
        row=json.loads(line)
        if row.get('mirrored') is False and row.get('motion_id') in by_id:
            sid=row['motion_id']
            if sid in alignment:raise ValueError('duplicate SSM alignment ID')
            alignment[sid]=row
    if set(alignment)!=set(by_id):raise ValueError('SSM evidence missing alignment')
    cases=[]
    for sid in sorted(by_id):
        row=alignment[sid];pin=by_id[sid]
        cases.append(audit_ssm_source(samples[sid],row,receipt_path=row['source_receipt'],receipt_sha256=pin.get('receipt_sha256')))
    if digest(index)!=index_hash or digest(manifest)!=manifest_hash:raise RuntimeError('SSM evidence changed during audit')
    return {'marker_evidence_cases':len(cases),'samples_without_marker_evidence':[r['sample_id'] for r in records if r['sample_id'] not in by_id],
            'alignment_sha256':index_hash,'evidence_manifest_sha256':manifest_hash,'cases':cases,'training_ready':False,
            'limitations':['Only local source/reference marker equality is checked, not true capture timing or official intermediate.',
                           'No overall source, licensing, subject, physical or training acceptance.']}

"""Validate binding of declared semantic reviews to caption and feature bytes.

This checks receipt structure and hashes, not the truth of the visual judgment,
source, license or physical quality. It cannot authorize training by itself.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re


def _digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _file(root,name,expected):
    if not isinstance(name,str) or not name or Path(name).is_absolute():raise ValueError('receipt path must be relative')
    if not isinstance(expected,str) or re.fullmatch(r'[0-9a-f]{64}',expected) is None:raise ValueError('invalid receipt SHA-256')
    p=(root/name).resolve(strict=True)
    if not p.is_relative_to(root) or not p.is_file() or _digest(p)!=expected:raise ValueError('receipt path or hash mismatch')
    return p


def audit_caption_receipts(records,*,evidence_root=None,evidence_manifest=None):
    if not isinstance(records,list) or not records:raise ValueError('records must be nonempty')
    accepted=[r for r in records if r.get('caption_accepted') is True]
    for r in records:
        if r.get('caption_accepted') is not None and not isinstance(r.get('caption_accepted'),bool):
            raise ValueError('caption_accepted must be a boolean')
    if (evidence_root is None)!=(evidence_manifest is None):raise ValueError('caption evidence root and manifest must be supplied together')
    if evidence_root is None:
        if accepted:raise ValueError('declared accepted captions require bound review receipts')
        return {'verified_semantic_receipts':0,'unaccepted_captions':len(records),'training_ready':False,
                'limitations':['Review receipt binding is not an independent semantic or physical assessment.']}
    root=Path(evidence_root).resolve(strict=True); mp=Path(evidence_manifest).resolve(strict=True)
    if not mp.is_file():raise ValueError('caption evidence manifest must be a file')
    manifest_hash=_digest(mp); bundle=json.loads(mp.read_text(encoding='utf8'))
    if not isinstance(bundle.get('receipts'),list):raise ValueError('caption receipt list required')
    by_id={}
    for row in bundle['receipts']:
        sid=row.get('sample_id')
        if not isinstance(sid,str) or not sid or sid in by_id:raise ValueError('duplicate or missing receipt ID')
        by_id[sid]=row
    if len(by_id)!=len(accepted) or set(by_id)!={r.get('sample_id') for r in accepted}:
        raise ValueError('receipt coverage must match declared accepted captions exactly')
    verified=[]
    for row in accepted:
        evidence=by_id[row['sample_id']]
        receipt=_file(root,evidence.get('receipt_path'),evidence.get('receipt_sha256'))
        audit=_file(root,evidence.get('independent_audit_path'),evidence.get('independent_audit_sha256'))
        if row.get('caption_review_sha256')!=evidence['receipt_sha256']:raise ValueError('manifest does not pin receipt')
        review=json.loads(receipt.read_text(encoding='utf8')); check=json.loads(audit.read_text(encoding='utf8'))
        if (review.get('sample_id')!=row['sample_id'] or review.get('caption')!=row.get('caption') or
            review.get('candidate_sha256')!=row.get('feature_sha256') or
            review.get('manual_semantic_assessment',{}).get('semantic_accepted_for_exported_samples') is not True or
            review.get('training_ready') is not False or check.get('training_ready') is not False or
            check.get('receipt_sha256')!=evidence['receipt_sha256']):
            raise ValueError('semantic receipt or independent audit does not bind this sample')
        assessment=review['manual_semantic_assessment']
        if not all(isinstance(assessment.get(k),str) and assessment[k].strip() for k in ('reviewer','reason','status')):
            raise ValueError('review must describe reviewer, scope and reason')
        sample=review.get('sample_range_s'); source=review.get('source_range_s')
        if (not isinstance(sample,list) or not isinstance(source,list) or len(sample)!=2 or len(source)!=2 or
            any(isinstance(x,bool) or not isinstance(x,(int,float)) for x in sample+source)):
            raise ValueError('valid review timing required')
        import math
        if not all(math.isfinite(x) for x in sample+source) or not source[0]<=sample[0]<sample[1]<=source[1]:
            raise ValueError('reviewed interval must be inside source bounds')
        verified.append({'sample_id':row['sample_id'],'receipt_sha256':evidence['receipt_sha256'],
                         'audit_sha256':evidence['independent_audit_sha256']})
        if _digest(receipt)!=evidence['receipt_sha256'] or _digest(audit)!=evidence['independent_audit_sha256']:
            raise RuntimeError('caption evidence changed while checking')
    if _digest(mp)!=manifest_hash:raise RuntimeError('caption manifest changed while checking')
    return {'verified_semantic_receipts':len(verified),'unaccepted_captions':len(records)-len(verified),
            'evidence_manifest_sha256':manifest_hash,'verified':verified,'training_ready':False,
            'limitations':['Checks structural and hash bindings only, not truth of visual review.',
                           'Does not verify source identity, licensing, physical quality or overall training readiness.']}

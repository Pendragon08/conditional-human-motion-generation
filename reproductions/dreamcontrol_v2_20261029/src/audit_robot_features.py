"""Audit robot feature manifests and train-only statistics, not physical quality.

Requires explicit group/source/caption metadata, 20 FPS, all three splits and
content hashes. Recomputes statistics from train frames and 22 decoded points.
Declared source groups and FPS are not independently proven by this check;
collision, contact, retargeting and licensing require separate evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from grouped_split import audit_assignments
from provenance_groups import audit_provenance


class Moments:
    def __init__(self, width):
        self.n=0; self.mean=np.zeros(width,dtype=np.float64); self.m2=np.zeros(width,dtype=np.float64)
    def add(self, x):
        x=np.asarray(x,dtype=np.float64)
        n=len(x); avg=x.mean(axis=0); m2=((x-avg)**2).sum(axis=0)
        delta=avg-self.mean; total=self.n+n
        self.m2 += m2 + delta**2*self.n*n/total
        self.mean += delta*n/total; self.n=total
    def values(self, minimum):
        if not self.n: raise ValueError('no training frames')
        return self.mean,np.maximum(np.sqrt(np.maximum(self.m2/self.n,0)),minimum)


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def audit(root, records, stats, *, min_std, decode_points=None):
    if not isinstance(records,list) or not records: raise ValueError('nonempty manifest records required')
    if not np.isfinite(min_std) or min_std<=0: raise ValueError('invalid standard deviation floor')
    audit_assignments(records)
    has_provenance = ['provenance_keys' in rec for rec in records]
    if any(has_provenance) and not all(has_provenance):
        raise ValueError('provenance_keys must be supplied for all records or none')
    provenance = audit_provenance(records) if all(has_provenance) else None
    if provenance is not None and not provenance['source_groups_disjoint']:
        raise ValueError('explicit provenance relationships cross splits')
    root=Path(root).resolve(strict=True)
    if decode_points is None:
        from data_loaders.humanml.scripts.motion_process import recover_from_ric
        def decode_points(x):
            with torch.no_grad(): return recover_from_ric(torch.from_numpy(x),22).numpy()
    counts=Counter(); caption_status=Counter(); content_splits={}; source_splits={}; feature=Moments(263); spatial=Moments(66)
    frames=0
    for rec in records:
        caption_accepted=rec.get('caption_accepted')
        if caption_accepted is not None and not isinstance(caption_accepted,bool):
            raise ValueError('caption_accepted must be boolean when present')
        caption_status['declared_accepted' if caption_accepted is True else 'unaccepted_or_unspecified']+=1
        for key in ('caption','caption_source','source_id','feature_path','feature_sha256'):
            if not isinstance(rec.get(key),str) or not rec[key].strip():
                raise ValueError(f'missing {key} for {rec.get("sample_id")}')
        fps=rec.get('fps')
        if isinstance(fps,bool) or not isinstance(fps,(int,float)) or not np.isfinite(fps) or abs(fps-20.)>1e-8:
            raise ValueError('each feature sequence must declare 20 FPS')
        if source_splits.setdefault(rec['source_id'],rec['split']) != rec['split']:
            raise ValueError('same declared source crosses splits')
        relative=Path(rec['feature_path'])
        if relative.is_absolute(): raise ValueError('feature path must be relative')
        path=(root/relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file() or path.suffix!='.npz': raise ValueError('invalid feature path')
        if len(rec['feature_sha256']) != 64 or sha256_file(path)!=rec['feature_sha256'].lower():
            raise ValueError(f'feature hash mismatch: {rec["sample_id"]}')
        with np.load(path,allow_pickle=False) as data:
            raw=np.asarray(data['features'])
            if not np.issubdtype(raw.dtype,np.number) or np.iscomplexobj(raw):
                raise ValueError(f'features must contain real numeric values: {rec["sample_id"]}')
            x=np.asarray(raw,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=263 or not 2<=len(x)<=199 or not np.isfinite(x).all():
            raise ValueError(f'invalid feature array: {rec["sample_id"]}')
        if not np.logical_or(np.isclose(x[:,-4:],0,atol=1e-6),np.isclose(x[:,-4:],1,atol=1e-6)).all():
            raise ValueError(f'foot contact features must be binary: {rec["sample_id"]}')
        h=hashlib.sha256(); h.update(np.asarray(x.shape,dtype='<i8').tobytes()); h.update(np.ascontiguousarray(x,dtype='<f4').tobytes())
        digest=h.hexdigest(); previous=content_splits.setdefault(digest,rec['split'])
        if previous!=rec['split']: raise ValueError('identical motion features cross splits')
        counts[rec['split']]+=1
        if rec['split']=='train':
            points=np.asarray(decode_points(x),dtype=np.float64)
            if points.shape!=(len(x),22,3) or not np.isfinite(points).all():
                raise ValueError(f'invalid decoded joints: {rec["sample_id"]}')
            feature.add(x); spatial.add(points.reshape(len(x),66)); frames+=len(x)
    if set(counts)!={'train','val','test'}: raise ValueError('train, val and test splits are required')
    fmean,fstd=feature.values(min_std); smean,sstd=spatial.values(min_std)
    for key,expected in [('feature_mean',fmean),('feature_std',fstd),('spatial_mean',smean),('spatial_std',sstd)]:
        actual=np.asarray(stats[key],dtype=np.float64).reshape(-1)
        if actual.shape!=expected.shape or not np.isfinite(actual).all() or not np.allclose(actual,expected,rtol=1e-5,atol=1e-6):
            raise ValueError(f'{key} does not match recomputed train-only statistics')
    return {'technical_checks_passed':True,'training_ready':False,'counts':dict(counts),'train_frames':frames,
            'provenance_components':provenance['component_count'] if provenance is not None else None,
            'explicit_provenance_groups_disjoint':provenance['source_groups_disjoint'] if provenance is not None else None,
            'caption_status_counts':dict(caption_status),
            'min_std':float(min_std),'exact_cross_split_duplicates':0,
            'limitations':['Declared fps and source grouping are not independently proven.',
                           'Caption acceptance is a declaration and is not independently verified.',
                           'Physics, foot contact, collision, retargeting, license and caption semantics are not verified.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--manifest',required=True,type=Path,help='JSON with a records array')
    parser.add_argument('--stats',required=True,type=Path,help='NPZ with feature_mean/std and spatial_mean/std')
    parser.add_argument('--min-std',required=True,type=float)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    manifest_hash=sha256_file(args.manifest); stats_hash=sha256_file(args.stats)
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    with np.load(args.stats,allow_pickle=False) as f:
        stats={key:f[key].copy() for key in ('feature_mean','feature_std','spatial_mean','spatial_std')}
    result=audit(args.root,manifest['records'],stats,min_std=args.min_std)
    # Detect changes during the audit before publishing its result.
    if sha256_file(args.manifest)!=manifest_hash or sha256_file(args.stats)!=stats_hash:
        raise RuntimeError('manifest or statistics changed during audit')
    result.update({'manifest_sha256':manifest_hash,'stats_sha256':stats_hash})
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False); f.write('\n')
    print(json.dumps({'technical_checks_passed':True,'training_ready':False,'counts':result['counts']}))


if __name__=='__main__': main()

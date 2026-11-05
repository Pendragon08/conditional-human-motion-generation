"""Check narrowly-scoped machine captions against hashed G1 FK positions.

This validates an ankle-height description only. It does not verify source
identity, licensing, time ground truth, contact, dynamics, or train readiness.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def audit_geometry_captions(records, *, evidence_manifest=None, evidence_sha256=None, state_root=None):
    if not isinstance(records,list):raise ValueError('records must be a list')
    supplied=(evidence_manifest,evidence_sha256,state_root)
    if not records:
        if any(x is not None for x in supplied):raise ValueError('geometry evidence without geometry captions')
        return {'status':'no_geometry_captions','cases':[],'geometry_checks_passed':False,'training_ready':False}
    if any(x is None for x in supplied):raise ValueError('geometry captions require evidence path, hash and state root')
    evidence_path=Path(evidence_manifest).resolve(strict=True); root=Path(state_root).resolve(strict=True)
    if not root.is_dir() or not isinstance(evidence_sha256,str) or _sha(evidence_path)!=evidence_sha256:
        raise ValueError('geometry evidence hash or state root invalid')
    evidence=[json.loads(x) for x in evidence_path.read_text(encoding='utf8').splitlines() if x.strip()]
    index={(r['sample_id'],r['candidate_caption']):r for r in evidence}
    if len(index)!=len(evidence) or len({r['sample_id'] for r in records})!=len(records):raise ValueError('duplicate geometry sample')
    cases=[]
    for row in records:
        sid=row['sample_id']; entry=index.get((sid,row.get('caption'))); origin=row.get('caption_origin')
        if entry is None or origin not in ('machine_geometry_v1','machine_geometry_v2') or row.get('caption_accepted') is not True:
            raise ValueError('missing candidate or explicit geometry caption origin')
        if (entry.get('caption_origin')!=origin or entry.get('caption_accepted') is not False or
            entry.get('training_ready') is not False or entry.get('feature_sha256')!=row.get('feature_sha256') or
            entry.get('candidate_caption')!=row.get('caption')):raise ValueError('candidate binding mismatch')
        rel=Path(entry['state_path'])
        if rel.is_absolute():raise ValueError('state path must be relative')
        path=(root/rel).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file() or _sha(path)!=entry.get('state_sha256'):
            raise ValueError('state hash or path invalid')
        with np.load(path,allow_pickle=False) as f:points=f['points_z_up'].copy();times=f['sample_times'].copy()
        n=len(points)
        if (points.shape!=(n,22,3) or n<15 or not np.isfinite(points).all() or
            times.shape!=(n,) or not np.allclose(times,np.arange(n)/20,atol=1e-10,rtol=0) or
            (origin=='machine_geometry_v1' and (entry.get('state_frame_count')!=n or entry.get('feature_frame_count')!=n-1))):
            raise ValueError('invalid state geometry or nominal time')
        ev=entry['evidence'];joint=ev.get('joint_index')
        specs={('machine_geometry_v1',7):('The humanoid raises and lowers its left ankle.',1),
               ('machine_geometry_v1',8):('The humanoid raises and lowers its right ankle.',1),
               ('machine_geometry_v2',20):('The humanoid raises and lowers its left wrist.',1),
               ('machine_geometry_v2',21):('The humanoid raises and lowers its right wrist.',1),
               ('machine_geometry_v2',0):('The humanoid lowers and raises its pelvis.',-1)}
        spec=specs.get((origin,joint))
        if (spec is None or row['caption']!=spec[0] or ev.get('window_frames')!=5 or ev.get('threshold_m')!=.12 or
            (origin=='machine_geometry_v2' and ev.get('direction')!=spec[1])):raise ValueError('unsupported geometry caption')
        direction=spec[1];curve=[direction*sum(float(points[i+d,joint,2]) for d in range(5))/5 for i in range(n-4)]
        peak=max(curve); k=curve.index(peak);rise=peak-curve[0];fall=peak-curve[-1]
        if k<4 or k>len(curve)-5 or rise<.12 or fall<.12:raise ValueError('ankle motion predicate failed')
        centers=[2,k+2,n-3]
        height_key='mean_z_m' if origin=='machine_geometry_v1' else 'oriented_mean_z_m'
        if (ev.get('center_indices')!=centers or not np.allclose(ev['nominal_seconds'],times[centers],rtol=0,atol=1e-10) or
            not np.allclose(ev[height_key]+[ev['rise_m'],ev['fall_m']],
                            [curve[0],peak,curve[-1],rise,fall],rtol=0,atol=1e-7)):
            raise ValueError('ankle motion evidence mismatch')
        if _sha(path)!=entry['state_sha256']:raise RuntimeError('state changed during geometry check')
        cases.append({'sample_id':sid,'joint_index':joint,'state_sha256':entry['state_sha256']})
    if _sha(evidence_path)!=evidence_sha256:raise RuntimeError('geometry evidence changed')
    return {'status':'geometry_caption_binding_checked','geometry_checks_passed':True,
            'evidence_sha256':evidence_sha256,'cases':cases,'training_ready':False,
            'limitations':['Joint height is based on FK positions and a five-frame average, not contact.',
                           'The label is machine-generated, not an original source caption.',
                           'This check does not authorize training or verify source, license, or physical quality.']}

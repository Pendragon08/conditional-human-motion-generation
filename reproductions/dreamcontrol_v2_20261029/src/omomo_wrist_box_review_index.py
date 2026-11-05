"""Index wrist-origin distances to the official box enclosing volume.

This is a review aid, not a hand contact, semantic or physical quality test.
Only original-size OMOMO robot tracks are included. The robot points and box
pose are the previously exported approximate 20 Hz states; the AABB of the
official visual mesh is a conservative enclosure, not collision geometry.
"""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MANIFEST=ROOT/'omomo_mixed_encoding_preview_20261101/manifest.json'
MESH=ROOT/'omniretarget_largebox_asset_20261101/largebox.obj'
MESH_SHA='dbcc11281f62e9226f49165252375080d2e490e5a3f0ab6ba917acbd8f7abc1c'
HERE=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def bounds(path):
    vertices=[]
    with path.open() as f:
        for line in f:
            if line.startswith('v '):
                v=line.split()
                if len(v)<4:raise ValueError('bad mesh vertex')
                vertices.append([float(x) for x in v[1:4]])
    x=np.asarray(vertices,dtype=np.float64)
    if x.shape!=(13551,3) or not np.isfinite(x).all():raise ValueError('invalid mesh vertices')
    return x.min(axis=0),x.max(axis=0)

def distances(points,obj,lo,hi):
    if (points.ndim!=3 or points.shape[1:]!=(22,3) or obj.shape!=(len(points),7)
        or not np.isfinite(points).all() or not np.isfinite(obj).all()):raise ValueError('invalid state')
    norm=np.linalg.norm(obj[:,:4],axis=1)
    if np.any(norm<1e-9) or not np.allclose(norm,1.,atol=1e-5,rtol=0):raise ValueError('bad object quaternion')
    rotation=Rotation.from_quat(obj[:,[1,2,3,0]]).as_matrix()
    delta=points[:,[20,21],:]-obj[:,None,4:]
    local=np.einsum('tji,tkj->tki',rotation,delta)
    excess=np.maximum(np.maximum(lo-local,local-hi),0.)
    return np.linalg.norm(excess,axis=-1)

def main():
    if digest(MESH)!=MESH_SHA:raise ValueError('box asset hash mismatch')
    lo,hi=bounds(MESH); mh=digest(MANIFEST)
    records=json.loads(MANIFEST.read_text())['records']
    originals=[r for r in records if r['source_member'].endswith('_original.npz')]
    if len(originals)!=277 or len({x['sample_id'] for x in originals})!=277:raise ValueError('unexpected coverage')
    total_frames=0; total_inside=np.zeros(2,dtype=int); min_dist=[]
    with (HERE/'records.jsonl').open('x',encoding='utf8') as out:
        for i,r in enumerate(originals):
            if i%10==0 and shutil.disk_usage(ROOT).free < 8*1024**3:raise RuntimeError('disk stop line')
            p=(ROOT/r['state_path']).resolve(strict=True)
            if not p.is_relative_to(ROOT) or digest(p)!=r['state_sha256']:raise ValueError('state hash mismatch')
            with np.load(p,allow_pickle=False) as f:
                points=np.asarray(f['points_z_up'],dtype=np.float64)
                obj=np.asarray(f['object_wxyz_xyz'],dtype=np.float64)
                time=np.asarray(f['sample_times'],dtype=np.float64)
            if time.shape!=(len(points),) or not np.isfinite(time).all() or np.any(np.diff(time)<=0):raise ValueError('bad timeline')
            d=distances(points,obj,lo,hi); k=np.argmin(d,axis=0); n=len(d)
            row={'sample_id':r['sample_id'],'caption':r['caption'],'split':r['auxiliary_split'],
                 'state_sha256':r['state_sha256'],'feature_sha256':r['feature_sha256'],'frames':n,
                 'min_distance_m':d[k,np.arange(2)].tolist(),'min_frame':k.tolist(),
                 'median_distance_m':np.median(d,axis=0).tolist(),
                 'wrist_inside_enclosure_frames':(d==0).sum(axis=0).tolist(),
                 'nominal_time_at_min_s':time[k].tolist(),'training_ready':False}
            out.write(json.dumps(row,allow_nan=False)+'\n');total_frames+=n;total_inside+=(d==0).sum(axis=0)
            min_dist.append(row['min_distance_m'])
            if digest(p)!=r['state_sha256']:raise RuntimeError('state changed during reading')
    if digest(MANIFEST)!=mh or digest(MESH)!=MESH_SHA:raise RuntimeError('input changed')
    summary={'manifest_sha256':mh,'mesh_sha256':MESH_SHA,'box_bounds_min_m':lo.tolist(),
             'box_bounds_max_m':hi.tolist(),'records_sha256':digest(HERE/'records.jsonl'),
             'script_sha256':digest(Path(__file__)),'samples':len(originals),'frames':total_frames,
             'wrist_inside_enclosure_frames':total_inside.tolist(),
             'sample_min_distance_median_m':np.median(min_dist,axis=0).tolist(),
             'training_ready':False,
             'limitations':['Points are wrist joint origins, not hand surfaces; distance to enclosure is not contact distance.',
              'No threshold, automatic caption verdict or rejection; inside enclosure is not collision proof.',
              'Source geometry convention, retargeting, time, identity, licensing and dynamics remain unaccepted.',
              'No augmented samples; box scale for those is unverified.']}
    with (HERE/'report.json').open('x',encoding='utf8') as out:json.dump(summary,out,indent=2,allow_nan=False)
    print(json.dumps(summary,allow_nan=False))

if __name__=='__main__':main()

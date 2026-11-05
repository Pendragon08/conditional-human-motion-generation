"""Independently recompute every wrist/enclosing-box review-index value.

Uses explicit scalar quaternion algebra rather than the generation program's
SciPy rotation. State points and object poses are shared inputs; their FK and
physical correctness are not independently evaluated.
"""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'omomo_wrist_box_review_index_20261101'
MANIFEST=ROOT/'omomo_mixed_encoding_preview_20261101/manifest.json'
MESH=ROOT/'omniretarget_largebox_asset_20261101/largebox.obj'
HERE=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    report=json.loads((BASE/'report.json').read_text())
    if digest(BASE/'records.jsonl')!=report['records_sha256'] or digest(MANIFEST)!=report['manifest_sha256'] or digest(MESH)!=report['mesh_sha256']:
        raise ValueError('index input hash mismatch')
    records={r['sample_id']:r for r in json.loads(MANIFEST.read_text())['records'] if r['source_member'].endswith('_original.npz')}
    rows=[json.loads(x) for x in (BASE/'records.jsonl').read_text().splitlines()]
    if len(rows)!=277 or len(records)!=277 or {r['sample_id'] for r in rows}!=set(records):raise ValueError('coverage mismatch')
    vertices=[]
    with MESH.open() as f:
        for line in f:
            if line.startswith('v '):vertices.append([float(v) for v in line.split()[1:4]])
    lo=np.min(vertices,axis=0);hi=np.max(vertices,axis=0)
    max_error=0.; frames=0
    for row in rows:
        rec=records[row['sample_id']];p=(ROOT/rec['state_path']).resolve(strict=True)
        if not p.is_relative_to(ROOT) or digest(p)!=row['state_sha256'] or row['state_sha256']!=rec['state_sha256']:
            raise ValueError('state binding mismatch')
        if row['caption']!=rec['caption'] or row['feature_sha256']!=rec['feature_sha256'] or row['split']!=rec['auxiliary_split']:
            raise ValueError('record binding mismatch')
        with np.load(p,allow_pickle=False) as f:
            points=np.asarray(f['points_z_up'],dtype=float)[:,[20,21]]
            obj=np.asarray(f['object_wxyz_xyz'],dtype=float); t=np.asarray(f['sample_times'],dtype=float)
        q=obj[:,:4]/np.linalg.norm(obj[:,:4],axis=1)[:,None]
        w,x,y,z=q.T
        # Quaternion R transpose times world-space wrist displacement.
        mat=np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                      2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                      2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=1).reshape(-1,3,3)
        delta=points-obj[:,None,4:];local=np.empty_like(delta)
        for axis in range(3):
            local[:,:,axis]=sum(mat[:,j,axis,None]*delta[:,:,j] for j in range(3))
        d=np.sqrt(np.square(np.maximum(np.maximum(lo-local,local-hi),0)).sum(axis=2))
        indices=np.argmin(d,axis=0);values=d[indices,np.arange(2)]
        comparisons=[(values,row['min_distance_m']),(np.median(d,axis=0),row['median_distance_m']),
                     (t[indices],row['nominal_time_at_min_s'])]
        for actual,expected in comparisons:max_error=max(max_error,float(np.max(np.abs(actual-np.asarray(expected)))))
        if row['frames']!=len(d) or indices.tolist()!=row['min_frame'] or (d==0).sum(axis=0).tolist()!=row['wrist_inside_enclosure_frames']:
            raise ValueError('frame count, minimum indices or zero counts mismatch')
        if digest(p)!=row['state_sha256']:raise RuntimeError('state changed during audit')
        frames+=len(d)
    if max_error>1e-10 or frames!=report['frames']:raise ValueError('aggregate mismatch')
    result={'records_sha256':report['records_sha256'],'manifest_sha256':report['manifest_sha256'],
            'mesh_sha256':report['mesh_sha256'],'script_sha256':digest(Path(__file__)),
            'samples':len(rows),'frames':frames,'max_absolute_difference':max_error,'training_ready':False,
            'limitations':['Same states and robot FK as generation; wrist origins and box enclosure only.',
                           'No contact, physical feasibility, caption semantics or training verdict.']}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))

if __name__=='__main__':main()

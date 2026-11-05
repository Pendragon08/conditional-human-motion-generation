"""Check mesh coordinate transforms over the seven review trajectories.

Hand-written quaternion matrix vs scipy; does not independently inspect
rendered pixels or establish physical contact.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PREVIEW=ROOT/'omomo_processed_full_preview_20261101';VIDEOS=ROOT/'omomo_box_review_20261101'
ASSET=ROOT/'omniretarget_largebox_asset_20261101';HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def manual_matrix(q):
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def main():
    review=json.loads((VIDEOS/'report.json').read_text()); manifest=PREVIEW/'manifest.json'
    if sha(manifest)!=review['preview_manifest_sha256'] or sha(ASSET/'largebox.obj')!=review['mesh_sha256']:
        raise ValueError('input provenance mismatch')
    records={r['sample_id']:r for r in json.loads(manifest.read_text())['records']}
    vertices=np.asarray(trimesh.load(ASSET/'largebox.obj',force='mesh',process=False).vertices,dtype=float)
    indices=sorted(set([int(np.argmin(vertices[:,i])) for i in range(3)]+[int(np.argmax(vertices[:,i])) for i in range(3)]))
    selected=vertices[indices];result=[];max_error=0.
    for case in review['cases']:
        row=records[case['sample_id']];path=(ROOT/row['state_path']).resolve(strict=True)
        if not path.is_relative_to(ROOT.resolve()) or sha(path)!=case['state_sha256']:raise ValueError('state mismatch')
        with np.load(path,allow_pickle=False) as f:obj=np.asarray(f['object_wxyz_xyz'],dtype=float)
        if obj.shape!=(case['frames'],7) or not np.isfinite(obj).all():raise ValueError('invalid object track')
        peak=0.
        for pose in obj:
            q=pose[:4]
            if abs(np.linalg.norm(q)-1)>1e-5:raise ValueError('invalid quaternion')
            a=selected@manual_matrix(q).T+pose[4:]
            b=Rotation.from_quat(q[[1,2,3,0]]).apply(selected)+pose[4:]
            peak=max(peak,float(np.max(np.abs(a-b))))
        max_error=max(max_error,peak)
        result.append({'sample_id':case['sample_id'],'frames':len(obj),'max_coordinate_error_m':peak})
    if max_error>1e-10:raise ValueError('coordinate transform mismatch')
    out={'video_report_sha256':sha(VIDEOS/'report.json'),'mesh_sha256':review['mesh_sha256'],
         'script_sha256':sha(__file__),'mesh_extreme_vertex_indices':indices,'cases':result,
         'frames':sum(r['frames'] for r in result),'max_coordinate_error_m':max_error,'training_ready':False,
         'limitations':['Same object pose inputs and selected vertices; no independent inspection of rendered pixels.',
                        'No motion semantics, surface contact, source license or physics acceptance.']}
    with (HERE/'report.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out))


if __name__=='__main__':main()

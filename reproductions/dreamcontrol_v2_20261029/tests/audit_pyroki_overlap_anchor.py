"""Recompute overlap, limits, source-keypoint and foot metrics independently."""
import hashlib,json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'foot_contact_fit_20261030'))
from g1_fk import URDFFK
from foot_geometry_audit import foot_spheres,sphere_centers
from waist_lock_ik import FIT_LINKS,_joint_limits

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(path,expected):
    if sha(path)!=expected:raise ValueError('window hash mismatch')
    with np.load(path,allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}
def overlap(a,b):
    common=np.intersect1d(a['frames'],b['frames']);i=np.searchsorted(a['frames'],common);j=np.searchsorted(b['frames'],common)
    if len(common)!=3:raise ValueError('expected three overlap frames')
    p=a['root_wxyz_xyz'];q=b['root_wxyz_xyz'];r0=Rotation.from_quat(p[i][:,[1,2,3,0]]);r1=Rotation.from_quat(q[j][:,[1,2,3,0]])
    return {'root_m':float(np.linalg.norm(p[i,4:]-q[j,4:],axis=1).max()),
            'root_rad':float((r0.inv()*r1).magnitude().max()),
            'joint_rad':float(np.abs(a['joint_pos'][i]-b['joint_pos'][j]).max())}
def main():
    base=ROOT/'pyroki_grab_s3_windowed_20261030';src=ROOT/'grab_s3_human_source_20261030/human_source.npz'
    doc=json.loads((HERE/'report.json').read_text());baseline=json.loads((base/'report.json').read_text());manifest=json.loads((HERE/'manifest.json').read_text())
    for p,key in ((src,'source_sha256'),(base/'report.json','baseline_report_sha256'),(HERE/'retarget_helpers/_utils.py','helper_sha256'),
                  (HERE/'pyroki_overlap_anchor_pilot.py','script_sha256'),(HERE/'windows.jsonl','rows_sha256')):
        if sha(p)!=doc[key]:raise ValueError('hash mismatch '+key)
    if doc['helper_sha256']!=manifest['anchor_helper_sha256'] or sha(base/'windows.jsonl')!=baseline['rows_sha256']:raise ValueError('manifest mismatch')
    rows=[json.loads(x) for x in (HERE/'windows.jsonl').read_text().splitlines()]
    base_rows=[json.loads(x) for x in (base/'windows.jsonl').read_text().splitlines()]
    if len(rows)!=5:raise ValueError('incomplete anchored windows')
    with np.load(src,allow_pickle=False) as d:human=d['human_joints'].copy();frames=d['input_frame_indices'].copy();fps=float(d['input_fps'])
    urdf=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
    if sha(urdf)!=doc['urdf_sha256'] or fps!=120.:raise ValueError('model/time mismatch')
    fk=URDFFK(urdf);offs,radii=foot_spheres(urdf); prior=load(base/'window_00.npz',doc['seed_artifact_sha256']); prior_base=prior
    records=[]
    for i,row in enumerate(rows,1):
        if row['window']!=i:raise ValueError('wrong order')
        anchored=load(HERE/f'window_{i:02d}.npz',row['artifact_sha256'])
        other=load(base/f'window_{i:02d}.npz',base_rows[i]['artifact_sha256'])
        if not np.array_equal(anchored['frames'],other['frames']) or anchored['frames'].tolist()!=row['frames'] or not np.array_equal(anchored['joint_names'],other['joint_names']):raise ValueError('not matched')
        raw_idx=np.searchsorted(frames,anchored['frames'])
        if not np.array_equal(frames[raw_idx],anchored['frames']):raise ValueError('source mismatch')
        rec={'window':i,'frames':anchored['frames'].tolist()}
        for label,w,prev in [('baseline',other,prior_base),('anchored',anchored,prior)]:
            p=w['root_wxyz_xyz'];q=w['joint_pos'];names=w['joint_names'].tolist()
            if p.shape!=(7,7) or q.shape!=(7,27) or not np.isfinite(p).all() or not np.isfinite(q).all():raise ValueError('bad output')
            norms=np.linalg.norm(p[:,:4],axis=1)
            if not np.allclose(norms,1,atol=1e-4,rtol=0):raise ValueError('bad quaternion')
            lo,hi=_joint_limits(urdf,names);violation=np.maximum(np.maximum(lo.numpy()-q,q-hi.numpy()),0.)
            centers=sphere_centers(fk,p[:,4:],p[:,[1,2,3,0]],q,names,offs)
            points=fk.forward(p[:,4:],p[:,[1,2,3,0]],q,names,links=FIT_LINKS)
            human_err=np.linalg.norm(points-human[raw_idx][:,[1,2,4,5,7,8,16,17,18,19,20,21]],axis=-1)
            rec[label]={'overlap':overlap(prev,w),'min_clearance_to_assumed_z0_m':float((centers[...,2]-radii).min()),
                'max_foot_horizontal_speed_mps':float((np.linalg.norm(np.diff(centers,axis=0)[...,:2],axis=-1)*fps).max()),
                'unscaled_world_keypoint_mean_m':float(human_err.mean()),'max_joint_limit_violation_rad':float(violation.max())}
        for k,kk in (('root_m','max_root_position_m'),('root_rad','max_root_angle_rad'),('joint_rad','max_joint_angle_rad')):
            if abs(rec['anchored']['overlap'][k]-row['overlap'][kk])>1e-7:raise ValueError('recorded overlap mismatch')
        for k in ('foot_min_clearance_m','max_foot_horizontal_speed_mps','unscaled_world_keypoint_mean_m'):
            key={'foot_min_clearance_m':'min_clearance_to_assumed_z0_m'}.get(k,k)
            if abs(rec['anchored'][key]-row[k])>1e-7:raise ValueError('recorded FK metric mismatch')
        records.append(rec);prior=anchored;prior_base=other
    agg={}
    for label in ('baseline','anchored'):
        v=[r[label] for r in records]
        agg[label]={'max_overlap_root_m':max(x['overlap']['root_m'] for x in v),
           'max_overlap_root_rad':max(x['overlap']['root_rad'] for x in v),
           'max_overlap_joint_rad':max(x['overlap']['joint_rad'] for x in v),
           'mean_min_foot_clearance_m':float(np.mean([x['min_clearance_to_assumed_z0_m'] for x in v])),
           'max_foot_speed_mps':max(x['max_foot_horizontal_speed_mps'] for x in v),
           'mean_unscaled_world_keypoint_error_m':float(np.mean([x['unscaled_world_keypoint_mean_m'] for x in v])),
           'max_joint_limit_violation_rad':max(x['max_joint_limit_violation_rad'] for x in v)}
    out={'source_sha256':sha(src),'report_sha256':sha(HERE/'report.json'),'script_sha256':sha(__file__),
         'aggregates':agg,'windows':records,
         'warning':'Five overlapping windows of one subject/action; baseline and anchor each compared to their own prior. Absolute keypoint errors ignore morphology; z=0 and source contact unknown. No object, physical dynamics or training validation; overlap anchor is experimental.'}
    with (HERE/'audit_report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(agg),flush=True)

if __name__=='__main__':main()

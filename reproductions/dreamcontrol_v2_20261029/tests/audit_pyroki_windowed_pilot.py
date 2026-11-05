"""Independently recheck fixed-window outputs and seam disagreement."""
import collections,hashlib,json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0,str(ROOT/'foot_contact_fit_20261030'))
from g1_fk import URDFFK
from foot_geometry_audit import foot_spheres,sphere_centers
from waist_lock_ik import _joint_limits

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    run=ROOT/'pyroki_grab_windowed_20261030';report=json.loads((run/'report.json').read_text())
    rows_path=run/'windows.jsonl';source=ROOT/'grab_human_robot_pilot_20261030/pilot_arrays.npz'
    urdf=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
    helper=ROOT/'pyroki_grab_temporal_20261030/retarget_helpers/_utils.py'
    for p,key in ((rows_path,'rows_sha256'),(source,'source_sha256'),(urdf,'urdf_sha256'),
                  (helper,'helper_sha256'),(run/'pyroki_grab_windowed_pilot.py','script_sha256')):
        if sha(p)!=report[key]:raise ValueError('hash mismatch '+key)
    rows=[json.loads(line) for line in rows_path.read_text().splitlines()]
    if len(rows)!=29 or report['window_count']!=29:raise ValueError('incomplete windows')
    with np.load(source,allow_pickle=False) as d:source_frames=d['input_frame_indices'].copy()
    prev=None;counts=collections.Counter();errors=[];seams=[];durations=[];qmax=0.;clearances=[];speeds=[]
    offsets,radii=foot_spheres(urdf);fk=URDFFK(urdf)
    for index,row in enumerate(rows):
        if row['window']!=index or row['artifact']!=f'window_{index:02d}.npz':raise ValueError('manifest mismatch')
        p=run/row['artifact']
        if sha(p)!=row['artifact_sha256']:raise ValueError('artifact mismatch')
        with np.load(p,allow_pickle=False) as d:
            frames=d['frames'].copy();fps=float(d['fps']);root=d['root_wxyz_xyz'].copy();joints=d['joint_pos'].copy();names=d['joint_names'].tolist()
        if fps!=120 or frames.tolist()!=row['frames'] or len(frames)!=7 or np.any(np.diff(frames)!=1):raise ValueError('wrong timing')
        if not np.isin(frames,source_frames).all() or root.shape!=(7,7) or joints.shape!=(7,27):raise ValueError('wrong frame or shape')
        if not np.isfinite(root).all() or not np.isfinite(joints).all():raise ValueError('nonfinite value')
        norm=np.linalg.norm(root[:,:4],axis=1);qmax=max(qmax,float(np.abs(norm-1).max()))
        if qmax>1e-4:raise ValueError('nonunit quaternion')
        lo,hi=_joint_limits(urdf,names);viol=np.maximum(np.maximum(lo.numpy()-joints,joints-hi.numpy()),0.)
        centers=sphere_centers(fk,root[:,4:],root[:,[1,2,3,0]],joints,names,offsets)
        c=float((centers[...,2]-radii[None,:,:]).min());s=float((np.linalg.norm(np.diff(centers,axis=0)[...,:2],axis=-1)*fps).max())
        if abs(c-row['foot_min_clearance_assumed_z0_m'])>1e-8 or abs(s-row['max_foot_horizontal_speed_mps'])>1e-7:raise ValueError('FK metric mismatch')
        clearances.append(c);speeds.append(s);durations.append(row['elapsed_s']);errors.append(float(viol.max()));counts.update(frames.tolist())
        if prev is not None:
            common=np.intersect1d(prev['frames'],frames);a=np.searchsorted(prev['frames'],common);b=np.searchsorted(frames,common)
            if not (3<=len(common)<=4):raise ValueError('wrong overlap')
            r0=Rotation.from_quat(prev['root'][a][:,[1,2,3,0]]);r1=Rotation.from_quat(root[b][:,[1,2,3,0]])
            seam={'frames':common.tolist(),'max_root_position_disagreement_m':float(np.linalg.norm(prev['root'][a,4:]-root[b,4:],axis=1).max()),
                  'max_root_rotation_disagreement_rad':float((r0.inv()*r1).magnitude().max()),
                  'max_joint_disagreement_rad':float(np.abs(prev['joints'][a]-joints[b]).max())}
            for k,v in seam.items():
                if k=='frames':
                    if v!=row['overlap_with_prior'][k]:raise ValueError('overlap frames mismatch')
                elif abs(v-row['overlap_with_prior'][k])>1e-7:raise ValueError('overlap metric mismatch')
            seams.append(seam)
        prev={'frames':frames,'root':root,'joints':joints}
    if sorted(counts)!=list(range(542,660)):raise ValueError('coverage mismatch')
    out={'source_sha256':sha(source),'report_sha256':sha(run/'report.json'),'rows_sha256':sha(rows_path),
         'script_sha256':sha(__file__),'windows_verified':len(rows),'unique_frames':len(counts),
         'first_frame':min(counts),'last_frame':max(counts),'fps':120,'repeated_frames':sum(v>1 for v in counts.values()),
         'max_joint_limit_violation_rad':max(errors),'max_quaternion_norm_error':qmax,
         'max_overlap_root_disagreement_m':max(s['max_root_position_disagreement_m'] for s in seams),
         'max_overlap_rotation_disagreement_rad':max(s['max_root_rotation_disagreement_rad'] for s in seams),
         'max_overlap_joint_disagreement_rad':max(s['max_joint_disagreement_rad'] for s in seams),
         'per_window_min_clearance_min_m':min(clearances),'per_window_min_clearance_max_m':max(clearances),
         'max_within_window_foot_horizontal_speed_mps':max(speeds),'first_window_elapsed_s':durations[0],
         'subsequent_window_median_elapsed_s':float(np.median(durations[1:])),
         'warning':'One action, unstitched windows, assumed plane, no object or dynamics. Overlap mismatch indicates separate solutions disagree, not necessarily error against ground truth. Runtime includes solving and is not a pure JIT compilation benchmark. Not training acceptance.'}
    with Path(__file__).with_name('audit_report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out),flush=True)

if __name__=='__main__':main()

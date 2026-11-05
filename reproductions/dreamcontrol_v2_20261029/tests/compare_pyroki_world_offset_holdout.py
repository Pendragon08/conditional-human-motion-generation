"""Independently compare frozen variant to matched S2/S3 baseline windows."""
import hashlib,json,re,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'foot_contact_fit_20261030'))
from g1_fk import URDFFK
from foot_geometry_audit import foot_spheres,sphere_centers
from waist_lock_ik import _joint_limits

LINKS=('pelvis_contour_link','left_hip_pitch_link','right_hip_pitch_link','left_knee_link','right_knee_link',
       'left_ankle_roll_link','right_ankle_roll_link','left_shoulder_roll_link','right_shoulder_roll_link',
       'left_elbow_link','right_elbow_link','left_wrist_yaw_link','right_wrist_yaw_link')
INDICES=(0,1,2,4,5,7,8,16,17,18,19,20,21)
SOURCES={'s2':'grab_human_robot_pilot_20261030/pilot_arrays.npz','s3':'grab_s3_human_source_20261030/human_source.npz'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def log_flags(p,n):
    parts=re.split(r'WINDOW (\d+) of \d+ elapsed [^\n]*',p.read_text())
    if len(parts)<2*n+1:raise ValueError('insufficient window logs')
    found=[]
    for i in range(n):
        if int(parts[2*i+1])!=i:raise ValueError('log order mismatch')
        m=re.findall(r'Terminated @ iteration #(\d+):[^\n]*criteria=\[([01 ]+)\]',parts[2*i])
        if len(m)!=1:raise ValueError('termination mismatch')
        found.append([int(x) for x in m[0][1].split()])
    return found

def main():
    manifest=json.loads((RUN/'manifest.json').read_text())
    if not (RUN/'COMPLETE.txt').is_file():raise ValueError('queue not complete')
    tuning=ROOT/'pyroki_s4_world_offset_variant_20261030/independent_comparison.json'
    if sha(tuning)!=manifest['fixed_s4_audit_sha256']:raise ValueError('tuning receipt changed')
    debug_helper=ROOT/'pyroki_s4_world_offset_variant_20261030/retarget_helpers/_utils.py';helper=RUN/'variant/retarget_helpers/_utils.py'
    original='    return transform, solution[var_joints], offset, solution[var_Ts_world_root]\n'
    replacement='    return transform, solution[var_joints]\n'
    if sha(debug_helper)!=manifest['fixed_debug_helper_sha256'] or debug_helper.read_text().replace(original,replacement)!=helper.read_text() or debug_helper.read_text().count(original)!=1:
        raise ValueError('frozen variant mismatch')
    if sha(helper)!=manifest['return_removed_helper_sha256'] or sha(RUN/'variant/report.json')!=manifest['variant_report_sha256']:
        raise ValueError('variant receipt mismatch')
    urdf=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
    fk=URDFFK(urdf);offs,radii=foot_spheres(urdf);all_results=[]
    for job in manifest['jobs']:
        subject=job['subject'];old=ROOT/job['baseline_dir'];new=RUN/subject;source=ROOT/SOURCES[subject]
        if sha(source)!=job['source_sha256'] or sha(new/'pyroki_grab_windowed_pilot.py')!=job['driver_sha256'] or sha(new/'audit_pyroki_windowed_pilot.py')!=job['audit_sha256']:
            raise ValueError('source or code changed')
        oldrep=json.loads((old/'report.json').read_text());newrep=json.loads((new/'report.json').read_text());newaudit=json.loads((new/'audit_report.json').read_text())
        if oldrep['source_sha256']!=job['source_sha256'] or newrep['source_sha256']!=job['source_sha256'] or newaudit['source_sha256']!=job['source_sha256']:
            raise ValueError('source receipt mismatch')
        if oldrep['weights']!=newrep['weights'] or newrep['helper_sha256']!=sha(helper) or newaudit['report_sha256']!=sha(new/'report.json') or newaudit['windows_verified']!=5:
            raise ValueError('weights or audit mismatch')
        for p,rep in ((old,oldrep),(new,newrep)):
            if sha(p/'windows.jsonl')!=rep['rows_sha256'] or sha(urdf)!=rep['urdf_sha256']:raise ValueError('rows/URDF changed')
        rows=[[json.loads(x) for x in (p/'windows.jsonl').read_text().splitlines()][:5] for p in (old,new)]
        if any(len(x)!=5 for x in rows):raise ValueError('incomplete matched windows')
        with np.load(source,allow_pickle=False) as d:human=d['human_joints'];src_frames=d['input_frame_indices'];fps=float(d['input_fps'])
        flags=[log_flags(old/'run.log',5),log_flags(new/'solver.log',5)];metrics=[[],[]];seams=[[],[]];previous=[None,None]
        for i in range(5):
            if rows[0][i]['window']!=i or rows[1][i]['window']!=i or rows[0][i]['frames']!=rows[1][i]['frames']:raise ValueError('frame mismatch')
            for k,folder in enumerate((old,new)):
                row=rows[k][i];artifact=folder/row['artifact']
                if sha(artifact)!=row['artifact_sha256']:raise ValueError('artifact hash mismatch')
                with np.load(artifact,allow_pickle=False) as d:
                    frame=d['frames'];root=d['root_wxyz_xyz'];q=d['joint_pos'];names=d['joint_names'].tolist()
                if frame.tolist()!=row['frames'] or len(frame)!=7:raise ValueError('artifact frames mismatch')
                ids=np.searchsorted(src_frames,frame)
                if not np.array_equal(src_frames[ids],frame):raise ValueError('input time mismatch')
                lo,hi=_joint_limits(urdf,names);viol=np.maximum(np.maximum(lo.numpy()-q,q-hi.numpy()),0.)
                points=fk.forward(root[:,4:],root[:,[1,2,3,0]],q,names,links=LINKS)
                err=np.linalg.norm(points-human[ids][:,INDICES,:],axis=-1)
                centers=sphere_centers(fk,root[:,4:],root[:,[1,2,3,0]],q,names,offs);gap=centers[...,2]-radii[None,:,:]
                speed=np.linalg.norm(np.diff(centers,axis=0)[...,:2],axis=-1)*fps
                metrics[k].append({'window':i,'flags':flags[k][i],'world_keypoint_mean_m':float(err.mean()),
                                   'min_sphere_clearance_m':float(gap.min()),'max_sphere_horizontal_speed_mps':float(speed.max()),
                                   'max_joint_limit_violation_rad':float(viol.max())})
                if previous[k] is not None:
                    prev=previous[k];common=np.intersect1d(prev['frames'],frame)
                    a=np.searchsorted(prev['frames'],common);b=np.searchsorted(frame,common)
                    q0=Rotation.from_quat(prev['root'][a][:,[1,2,3,0]]);q1=Rotation.from_quat(root[b][:,[1,2,3,0]])
                    seams[k].append({'root_m':float(np.linalg.norm(prev['root'][a,4:]-root[b,4:],axis=-1).max()),
                                     'angle_rad':float((q0.inv()*q1).magnitude().max()),'joint_rad':float(np.abs(prev['q'][a]-q[b]).max())})
                previous[k]={'frames':frame.copy(),'root':root.copy(),'q':q.copy()}
        summaries=[]
        for m,s in zip(metrics,seams):
            summaries.append({'mean_world_keypoint_m':float(np.mean([x['world_keypoint_mean_m'] for x in m])),
                              'mean_window_min_clearance_m':float(np.mean([x['min_sphere_clearance_m'] for x in m])),
                              'max_window_foot_speed_mps':float(max(x['max_sphere_horizontal_speed_mps'] for x in m)),
                              'max_joint_limit_violation_rad':float(max(x['max_joint_limit_violation_rad'] for x in m)),
                              'windows_without_stop_flags':sum(not any(x['flags']) for x in m),
                              'max_seam_root_m':float(max(x['root_m'] for x in s)),
                              'max_seam_angle_rad':float(max(x['angle_rad'] for x in s)),
                              'max_seam_joint_rad':float(max(x['joint_rad'] for x in s))})
        all_results.append({'subject':subject,'baseline_report_sha256':sha(old/'report.json'),'variant_report_sha256':sha(new/'report.json'),
                            'variant_audit_sha256':sha(new/'audit_report.json'),'baseline':summaries[0],'variant':summaries[1],
                            'baseline_windows':metrics[0],'variant_windows':metrics[1]})
    out={'script_sha256':sha(__file__),'manifest_sha256':sha(RUN/'manifest.json'),'helper_sha256':sha(helper),'urdf_sha256':sha(urdf),
         'results':all_results,'limitations':'Two subjects, five overlapping windows each; fixed experimental variant, approximate v1 solver, assumed z=0 and low-speed mask. Unscaled keypoint metrics and baseline differences are not ground truth. No inferred contact forces, Isaac Lab dynamics, author-v2 equivalence or training acceptance.'}
    with (RUN/'independent_comparison.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({'subjects':[x['subject'] for x in all_results]}),flush=True)

if __name__=='__main__':main()

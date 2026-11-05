"""Compare 3 already-computed human sources and PyRoki window audits.

Foot joint heights are not sole geometry; z=0 is an unverified assumption.
No GRAB contact or physical accuracy is inferred from this comparison.
"""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
CASES=(('s1_pour','grab_s1_pour_human_source_20261030/human_source.npz','pyroki_grab_s1_pour_windowed_20261030'),
       ('s2_fly','grab_human_robot_pilot_20261030/pilot_arrays.npz','pyroki_grab_windowed_20261030'),
       ('s3_eat','grab_s3_human_source_20261030/human_source.npz','pyroki_grab_s3_windowed_20261030'))

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
    return h.hexdigest()

def main():
    results=[]
    for name,source_rel,run_rel in CASES:
        source=ROOT/source_rel;run=ROOT/run_rel;receipt=json.loads((source.parent/'report.json').read_text());audit=json.loads((run/'audit_report.json').read_text());base=json.loads((run/'report.json').read_text())
        if sha(source)!=receipt['artifact_sha256'] or sha(source)!=audit['source_sha256'] or sha(run/'report.json')!=audit['report_sha256'] or sha(run/'windows.jsonl')!=audit['rows_sha256']:raise ValueError('input provenance mismatch '+name)
        if audit['windows_verified']!=29 or audit['unique_frames']!=118 or base['window_count']!=29:raise ValueError('incomplete window audit '+name)
        with np.load(source,allow_pickle=False) as z:
            human=np.asarray(z['human_joints'],float);frames=np.asarray(z['input_frame_indices']);fps=float(z['input_fps'])
        if human.shape!=(120,22,3) or not np.isfinite(human).all() or fps!=120 or not np.all(np.diff(frames)==1) or audit['first_frame']!=frames[2] or audit['last_frame']!=frames[-1]:raise ValueError('timing or sample mismatch '+name)
        feet=human[:,[10,11],:];speed=np.linalg.norm(np.diff(feet,axis=0)*fps,axis=2)
        metric=[]
        for i,side in enumerate(('left','right')):
            heights=feet[:,i,2];vel=speed[:,i]
            metric.append({'side':side,'source_foot_joint_z_min_m':float(heights.min()),'source_foot_joint_z_median_m':float(np.median(heights)),
                           'source_foot_joint_z_max_m':float(heights.max()),'source_foot_joint_speed_median_mps':float(np.median(vel)),
                           'source_foot_joint_speed_max_mps':float(vel.max()),'speed_below_0_05_mps_fraction':float(np.mean(vel<.05))})
        results.append({'case':name,'source_sha256':sha(source),'window_audit_sha256':sha(run/'audit_report.json'),
                        'source_frame_range':[int(frames[0]),int(frames[-1])],'source_declared_fps':fps,
                        'human_foot_joint_metrics':metric,
                        'g1_window_min_foot_sphere_clearance_range_m':[audit['per_window_min_clearance_min_m'],audit['per_window_min_clearance_max_m']],
                        'g1_max_within_window_foot_horizontal_speed_mps':audit['max_within_window_foot_horizontal_speed_mps'],
                        'g1_max_overlapping_root_disagreement_m':audit['max_overlap_root_disagreement_m']})
    out={'script_sha256':sha(Path(__file__)),'results':results,
         'warning':'Three separately retargeted one-second GRAB samples, not independent physical benchmarks. SMPL-X foot joints are not sole contact geometry. Source floor z=0 is unverified; foot speed is a heuristic, not measured contact. G1 values are independently solved short windows, not stitched trajectories or task dynamics; no training acceptance.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()

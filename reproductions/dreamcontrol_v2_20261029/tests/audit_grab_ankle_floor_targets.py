"""Inspect achieved ankle positions against the legacy z=0 floor targets.

The robot ankle origin is not its sole. Targets are proxies, not real contact.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0,str(ROOT/'grab_s3_object_wrist_audit_v2_20261030'))
from grab_s3_object_wrist_audit_v2 import chosen,load
sys.path.insert(0,str(ROOT/'foot_contact_fit_20261030'))
from g1_fk import URDFFK

CASES=(('s1_pour','grab_s1_pour_human_source_20261030/human_source.npz','pyroki_grab_s1_pour_windowed_20261030'),
       ('s2_fly','grab_human_robot_pilot_20261030/pilot_arrays.npz','pyroki_grab_windowed_20261030'),
       ('s3_eat','grab_s3_human_source_20261030/human_source.npz','pyroki_grab_s3_windowed_20261030'))

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    urdf=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf');fk=URDFFK(urdf);results=[]
    for case,source_rel,run_rel in CASES:
        source=ROOT/source_rel;run=ROOT/run_rel;receipt=json.loads((source.parent/'report.json').read_text());report=json.loads((run/'report.json').read_text());audit=json.loads((run/'audit_report.json').read_text())
        if sha(source)!=receipt['artifact_sha256'] or sha(source)!=report['source_sha256'] or sha(source)!=audit['source_sha256'] or sha(run/'report.json')!=audit['report_sha256'] or sha(run/'windows.jsonl')!=report['rows_sha256'] or sha(urdf)!=report['urdf_sha256']:raise ValueError('source changed '+case)
        rows=[json.loads(x) for x in (run/'windows.jsonl').read_text().splitlines()]
        if len(rows)!=29:raise ValueError('incomplete windows')
        windows=[load(run/x['artifact'],x['artifact_sha256']) for x in rows]
        frames,root,q=chosen(windows);names=windows[0]['joint_names'].tolist()
        with np.load(source,allow_pickle=False) as z:human=np.asarray(z['human_joints'],float);source_frames=z['input_frame_indices'].copy();fps=float(z['input_fps'])
        if human.shape!=(120,22,3) or fps!=120 or not np.all(np.diff(source_frames)==1) or not np.array_equal(frames,source_frames[2:]):raise ValueError('source alignment error')
        ix=np.searchsorted(source_frames,frames);robot=fk.forward(root[:,4:],root[:,[1,2,3,0]],q,names,links=('left_ankle_roll_link','right_ankle_roll_link'))
        if robot.shape!=(118,2,3) or not np.isfinite(robot).all():raise ValueError('invalid FK')
        metrics=[]
        for side,foot,group,j in (('left',10,[7,10],0),('right',11,[8,11],1)):
            # The original window driver foot_detect checks either ankle or toe.
            speed=np.linalg.norm((human[ix][:,group]-human[ix-1][:,group])*fps,axis=2)
            mask=np.any(speed<.05,axis=1)
            target=human[ix,foot,:].copy();target[:,2]=0.
            e=robot[:,j,:]-target
            if not mask.any():raise ValueError('no low-speed frames in '+case)
            metrics.append({'side':side,'low_speed_frames':int(mask.sum()),'selected_frames':len(frames),
                            'ankle_minus_z0_median_m':float(np.median(robot[mask,j,2])),
                            'ankle_minus_z0_range_m':[float(robot[mask,j,2].min()),float(robot[mask,j,2].max())],
                            'ankle_xy_target_error_median_m':float(np.median(np.linalg.norm(e[mask,:2],axis=1))),
                            'ankle_xyz_target_error_median_m':float(np.median(np.linalg.norm(e[mask],axis=1)))})
        results.append({'case':case,'source_sha256':sha(source),'window_audit_sha256':sha(run/'audit_report.json'),'feet':metrics})
    output={'urdf_sha256':sha(urdf),'script_sha256':sha(Path(__file__)),'results':results,
            'warning':'Central-window selections from three short independent-window trials using old approximate PyRoki objectives. Low speed is a heuristic. Robot ankle origin is not a sole contact; human foot point is not robot ankle. Assumed z=0 and input model are unverified. Distances indicate objective tradeoff, not physical truth, real contact or training acceptance.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf8') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps(output),flush=True)
if __name__=='__main__':main()

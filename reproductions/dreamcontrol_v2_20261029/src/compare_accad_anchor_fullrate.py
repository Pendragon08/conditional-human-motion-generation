"""Compare audited anchored and unanchored ACCAD windows on identical frames."""
import hashlib
import json
from pathlib import Path
import numpy as np
from g1_fk import URDFFK
from foot_geometry_audit import foot_spheres,sphere_centers

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'accad_013267_pyroki_fullrate_20261031'
ANCHOR=ROOT/'accad_013267_anchor_fullrate_20261031'
SOURCE=ROOT/'accad_013267_fullrate_20261031/human_120hz.npz'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE=Path(__file__).resolve().parent
LINKS=['left_hip_pitch_link','right_hip_pitch_link','left_knee_link','right_knee_link',
       'left_ankle_pitch_link','right_ankle_pitch_link','left_shoulder_pitch_link',
       'right_shoulder_pitch_link','left_elbow_link','right_elbow_link','left_wrist_yaw_link','right_wrist_yaw_link']
IDX=[1,2,4,5,7,8,16,17,18,19,20,21]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    with np.load(SOURCE,allow_pickle=False) as f:human=f['human'].copy(); source_frames=f['source_frames'].copy()
    fk=URDFFK(URDF);offsets,radii=foot_spheres(URDF);results=[]
    for folder in (BASE,ANCHOR):
        report=json.loads((folder/'report.json').read_text());audit=json.loads((folder/'audit_report.json').read_text())
        assert sha(folder/'report.json')==audit['report_sha256'] and sha(SOURCE)==audit['source_sha256']
        assert sha(URDF)==audit['urdf_sha256'] and audit['windows_verified']==88
        rows=[json.loads(x) for x in (folder/'windows.jsonl').read_text().splitlines()]
        assert sha(folder/'windows.jsonl')==report['rows_sha256']
        error_sum=0.;error_count=0;error_max=0.;clearance=[];foot_speed=[]
        for row in rows:
            p=folder/row['artifact'];assert sha(p)==row['artifact_sha256']
            with np.load(p,allow_pickle=False) as f:
                ix=f['sample_indices'].copy();root=f['root_wxyz_xyz'].copy();q=f['joint_pos'].copy()
                names=[str(x) for x in f['joint_names']];np.testing.assert_array_equal(f['source_frames'],source_frames[ix])
            actual=fk.forward(root[:,4:],root[:,[1,2,3,0]],q,names,links=LINKS)
            err=np.linalg.norm(actual-human[ix][:,IDX],axis=-1)
            error_sum+=float(err.sum());error_count+=err.size;error_max=max(error_max,float(err.max()))
            centers=sphere_centers(fk,root[:,4:],root[:,[1,2,3,0]],q,names,offsets)
            clearance.append(float((centers[:,:,:,2]-radii[None,:,:]).min()))
            foot_speed.append(float(np.linalg.norm(np.diff(centers,axis=0)[...,:2],axis=-1).max()*120))
        results.append({'report_sha256':sha(folder/'report.json'),'audit_sha256':sha(folder/'audit_report.json'),
                        'max_overlap_root_m':audit['max_overlap_root_m'],
                        'max_overlap_rotation_rad':audit['max_overlap_rotation_rad'],
                        'max_overlap_joint_rad':audit['max_overlap_joint_rad'],
                        'max_joint_limit_violation_rad':audit['max_joint_limit_violation_rad'],
                        'unscaled_world_keypoint_mean_m':error_sum/error_count,'unscaled_world_keypoint_max_m':error_max,
                        'per_window_min_foot_clearance_min_m':min(clearance),
                        'per_window_min_foot_clearance_max_m':max(clearance),
                        'within_window_max_foot_horizontal_speed_mps':max(foot_speed)})
    out={'baseline':results[0],'soft_anchor':results[1],'source_sha256':sha(SOURCE),'urdf_sha256':sha(URDF),
         'training_ready':False,'limitations':['Assumed floor z=0, source header 120Hz and static collision spheres.',
             'World keypoint differences include morphology and alignment; not truth error.',
             'Per-window metrics repeat overlapping frames and are not physical tracking or contact validation.',
             'One candidate and exploratory anchor weight.']}
    with (HERE/'comparison.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()

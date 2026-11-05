"""Descriptive whole-clip arm and ankle trajectories for a caption decision."""
import hashlib
import json
from pathlib import Path
import numpy as np
from g1_fk import URDFFK
from render_five_caption_review import human_data

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'six_family_disjoint_preview_v2_20261101'
VID=ROOT/'five_caption_review_videos_20261031'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def metrics(p):
    # Keep left/right separately; height is relative to each shoulder or initial ankle.
    wrist=np.stack([p[:,20,2]-p[:,16,2],p[:,21,2]-p[:,17,2]],axis=1)
    ankle_xy=np.linalg.norm(p[:,7,:2]-p[:,8,:2],axis=1)
    ankle_z=np.stack([p[:,7,2]-p[0,7,2],p[:,8,2]-p[0,8,2]],axis=1)
    return np.column_stack([wrist,ankle_xy,ankle_z])


def main():
    manifest=json.loads((BASE/'manifest.json').read_text())['records'];r=next(x for x in manifest if x['sample_id']=='003017')
    states=json.loads((BASE/'state_manifest.json').read_text())['records'];s=next(x for x in states if x['sample_id']=='003017')
    source=BASE/s['state_path'];assert sha(source)==s['state_sha256'] and sha(BASE/r['feature_path'])==r['feature_sha256']
    witness=json.loads((VID/'report.json').read_text());prior=next(x for x in witness['cases'] if x['sample_id']=='003017')
    assert sha(VID/'003017.mp4')==prior['video_sha256'] and sha(VID/'report.json')==json.loads((VID/'audit_report.json').read_text())['report_sha256']
    with np.load(source,allow_pickle=False) as f:
        root=f['root_wxyz_xyz'].copy();q=f['joint_pos'].copy();t=f['sample_times'].copy();names=f['joint_names'].tolist()
    robot=URDFFK(URDF).forward(root[:,4:],root[:,[1,2,3,0]],q,names)
    human,provenance=human_data('003017',t)
    assert len(t)==46 and human.shape==robot.shape==(46,22,3)
    hm=metrics(human);rm=metrics(robot)
    columns=['left_wrist_z_minus_shoulder_m','right_wrist_z_minus_shoulder_m',
             'ankle_horizontal_separation_m','left_ankle_z_minus_first_m','right_ankle_z_minus_first_m']
    out=HERE/'observables.npz'
    with out.open('xb') as f:np.savez_compressed(f,times=t,human=hm,robot=rm)
    report={'sample_id':'003017','state_sha256':sha(source),'feature_sha256':r['feature_sha256'],
        'human_source_evidence':provenance,'urdf_sha256':sha(URDF),'video_sha256':prior['video_sha256'],
        'observables_sha256':sha(out),'columns':columns,'samples':len(t),
        'human_ranges':np.ptp(hm,axis=0).tolist(),'robot_ranges':np.ptp(rm,axis=0).tolist(),
        'training_ready':False,'script_sha256':sha(Path(__file__)),
        'limitations':['No calibrated ground, contact, jumping force, true sample rate or physical acceptance.',
                       'Ankle height relative to first sample is not evidence of takeoff.',
                       'Fitted body and robot FK are approximate; no automatic caption acceptance.']}
    with (HERE/'report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))


if __name__=='__main__':main()

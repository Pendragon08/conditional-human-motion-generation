"""Describe a 20-frame GRAB/G1 window without claiming object contact."""
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OLD=ROOT/'grab_s3_weak_label_review_20261101';HUMAN=ROOT/'grab_s3_human_source_20261030'
BRIDGE=ROOT/'grab_robot_feature_bridge_20261031';HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    review=json.loads((OLD/'report.json').read_text());audit=json.loads((OLD/'audit.json').read_text())
    human_file=HUMAN/'human_source.npz';robot_file=BRIDGE/'s3_features.npz'
    assert sha(OLD/'report.json')==audit['report_sha256']
    assert sha(human_file)==review['human_sha256'] and sha(robot_file)==review['bridge_sha256']
    with np.load(human_file,allow_pickle=False) as f:h=f['human_joints'].astype(float);ids=f['input_frame_indices'].copy()
    with np.load(robot_file,allow_pickle=False) as f:frames=f['source_frames'].copy();robot=f['points_z_up_world'].astype(float)
    h=h[[int(np.nonzero(ids==x)[0].item()) for x in frames]]
    data={};summaries={}
    for kind,points in [('human',h),('robot',robot)]:
        summaries[kind]={}
        for side,shoulder,wrist in [('left',16,20),('right',17,21)]:
            height=points[:,wrist,2]-points[:,shoulder,2]
            dist=np.linalg.norm(points[:,wrist]-points[:,15],axis=1)
            data[kind+'_'+side+'_wrist_above_shoulder_m']=height
            data[kind+'_'+side+'_wrist_head_distance_m']=dist
            summaries[kind][side]={'first_wrist_height_m':float(height[0]),'last_wrist_height_m':float(height[-1]),
                'height_change_m':float(height[-1]-height[0]),'first_wrist_head_m':float(dist[0]),'last_wrist_head_m':float(dist[-1]),
                'max_height_frame_index':int(height.argmax()),'min_height_frame_index':int(height.argmin())}
    with (HERE/'series.npz').open('xb') as f:np.savez_compressed(f,source_frames=frames,**data)
    out={'source_review_sha256':sha(OLD/'report.json'),'source_audit_sha256':sha(OLD/'audit.json'),
         'human_sha256':sha(human_file),'robot_sha256':sha(robot_file),'series_sha256':sha(HERE/'series.npz'),
         'script_sha256':sha(Path(__file__)),'frames':len(frames),'first_last_source_frames':[int(frames[0]),int(frames[-1])],
         'summary':summaries,'caption_accepted':False,'training_ready':False,
         'limitations':['Joint locations use approximate fitted human and retargeted robot.',
                        'Head joint is not mouth; movement is not proof of eating, grasp or source physical contact.',
                        'No automatically generated caption is accepted and no source/time/physical gate is passed.']}
    with (HERE/'report.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()

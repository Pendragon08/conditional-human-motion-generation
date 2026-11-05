"""Independently bind the manual annotation and recalculate all 20 points."""
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent;HUMAN=ROOT/'grab_s3_human_source_20261030/human_source.npz'
FEATURE=ROOT/'grab_robot_feature_bridge_20261031/s3_features.npz'
MEASURE=ROOT/'grab_s3_arm_caption_candidate_20261101'
REVIEW=ROOT/'grab_s3_weak_label_review_20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    a=json.loads((HERE/'annotation.json').read_text());report=json.loads((REVIEW/'report.json').read_text())
    assert a['caption']=='a person lowers their right arm.' and 'manual visual annotation' in a['caption_origin']
    assert a['training_ready'] is False and sha(HUMAN)==a['human_sha256'] and sha(FEATURE)==a['feature_sha256']
    assert sha(REVIEW/'report.json')==a['review_sha256'] and sha(MEASURE/'report.json')==a['measure_sha256']
    assert sha(MEASURE/'series.npz')==a['measure_series_sha256']
    for x in report['figures']:assert sha(REVIEW/x['path'])==x['sha256']
    with np.load(HUMAN,allow_pickle=False) as f:h=f['human_joints'].astype(float);idx=f['input_frame_indices'].copy()
    with np.load(FEATURE,allow_pickle=False) as f:frames=f['source_frames'].copy();r=f['points_z_up_world'].astype(float);n=len(f['features'])
    assert frames.tolist()==a['source_frames'] and n==a['feature_rows']==19
    errors=[];delta={}
    with np.load(MEASURE/'series.npz',allow_pickle=False) as s:
        for name,points in [('human',h[[int(np.nonzero(idx==i)[0].item()) for i in frames]]),('robot',r)]:
            heights=[]
            for i in range(len(frames)):
                heights.append(float(points[i,21,2])-float(points[i,17,2]))
                errors.append(abs(heights[-1]-float(s[name+'_right_wrist_above_shoulder_m'][i])))
            delta[name]=heights[-1]-heights[0]
            assert abs(delta[name]-a['right_wrist_relative_to_shoulder_height_change_m'][name])<1e-12
            assert delta[name]<0
    out={'annotation_sha256':sha(HERE/'annotation.json'),'script_sha256':sha(Path(__file__)),
         'frames_checked':len(frames),'max_height_difference_m':max(errors),'height_change_m':delta,
         'training_ready':False,'limitations':'Checks same joint arrays and image hashes, not a second semantic reviewer or independently reconstructed poses.'}
    with (HERE/'audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()

"""Trace full-clip arm and hip geometry for the original punch caption."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
REVIEW=ROOT/'ssm_anchored_caption_review_20261031/report.json'
SOURCE=ROOT/'ssm_rawfps_human_batch_20261031/007410/human_rawfps.npz'
CANDIDATE=ROOT/'ssm_anchored_20hz_candidates_20261031/007410.npz'
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def metrics(p):
    arrays={}
    forward=((p[0,10,:2]-p[0,7,:2])+(p[0,11,:2]-p[0,8,:2]))/2
    if np.linalg.norm(forward)<1e-8:raise ValueError('degenerate initial toe direction')
    forward=forward/np.linalg.norm(forward)
    for side,shoulder,elbow,wrist in [('left',16,18,20),('right',17,19,21)]:
        upper=p[:,shoulder]-p[:,elbow];fore=p[:,wrist]-p[:,elbow]
        denom=np.linalg.norm(upper,axis=1)*np.linalg.norm(fore,axis=1)
        if np.any(denom<=1e-8):raise ValueError('degenerate arms')
        arrays[side+'_elbow_angle_deg']=np.degrees(np.arccos(np.clip((upper*fore).sum(axis=1)/denom,-1,1)))
        arrays[side+'_wrist_height_relative_shoulder_m']=p[:,wrist,2]-p[:,shoulder,2]
        arrays[side+'_reach_m']=np.linalg.norm(p[:,wrist]-p[:,shoulder],axis=1)
        arrays[side+'_wrist_initial_toe_axis_m']=(p[:,wrist,:2]-p[:,shoulder,:2])@forward
        arrays[side+'_wrist_height_relative_neck_m']=p[:,wrist,2]-p[:,12,2]
    vec=p[:,2,:2]-p[:,1,:2]
    if np.any(np.linalg.norm(vec,axis=1)<1e-8):raise ValueError('degenerate hips')
    angle=np.unwrap(np.arctan2(vec[:,1],vec[:,0]));arrays['hip_yaw_change_deg']=np.degrees(angle-angle[0])
    return arrays

def main():
    review=json.loads(REVIEW.read_text());item=next(x for x in review['cases'] if x['id']=='007410')
    assert sha(CANDIDATE)==item['input_sha256'] and sha(SOURCE)==item['source_sha256']
    caption=item['all_caption_records'][0]['caption']
    with np.load(SOURCE,allow_pickle=False) as f:st=f['source_times'].copy();h=f['human'].copy()
    with np.load(CANDIDATE,allow_pickle=False) as f:t=f['sample_times'].copy();robot=f['points_z_up_world'].copy()
    assert len(t)==78 and t[0]>=st[0] and t[-1]<=st[-1]
    human=np.column_stack([np.interp(t,st,h.reshape(len(st),66)[:,j]) for j in range(66)]).reshape(-1,22,3)
    series={};summary={}
    for label,poses in [('human',human),('robot',robot)]:
        m=metrics(poses);summary[label]={}
        for key,x in m.items():
            series[label+'_'+key]=x
            summary[label][key]={'min':float(x.min()),'min_frame':int(x.argmin()),'max':float(x.max()),'max_frame':int(x.argmax()),
                                 'first':float(x[0]),'last':float(x[-1])}
    with (HERE/'series.npz').open('xb') as f:np.savez_compressed(f,times=t,**series)
    out={'sample_id':'007410','frames':len(t),'caption':caption,'source_sha256':sha(SOURCE),
         'candidate_sha256':sha(CANDIDATE),'review_sha256':sha(REVIEW),'series_sha256':sha(HERE/'series.npz'),
         'script_sha256':sha(Path(__file__)),'summary':summary,'caption_accepted':False,'training_ready':False,
         'limitations':['Relative distances and angles describe a fitted skeleton and approximate G1, not intent or force.',
                        'Hip baseline depends on frame and joint conventions; no contact or independent clock.',
                        'Name and caption association do not prove official source identity.',
                        'No automatic caption or physical acceptance.']}
    with (HERE/'report.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))

if __name__=='__main__':main()

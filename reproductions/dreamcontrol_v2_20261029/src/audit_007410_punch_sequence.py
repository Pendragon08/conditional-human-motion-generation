"""Scalar re-evaluation of saved arm/hip measurements; no semantic verdict."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'ssm_rawfps_human_batch_20261031/007410/human_rawfps.npz'
CANDIDATE=ROOT/'ssm_anchored_20hz_candidates_20261031/007410.npz'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sub(a,b):return [float(a[i])-float(b[i]) for i in range(len(a))]
def norm(a):return math.sqrt(sum(v*v for v in a))

def main():
    report=json.loads((HERE/'report.json').read_text())
    assert sha(SOURCE)==report['source_sha256'] and sha(CANDIDATE)==report['candidate_sha256']
    assert sha(HERE/'series.npz')==report['series_sha256']
    with np.load(SOURCE,allow_pickle=False) as f:st=f['source_times'].astype(float);h=f['human'].astype(float)
    with np.load(CANDIDATE,allow_pickle=False) as f:t=f['sample_times'].astype(float);rob=f['points_z_up_world'].astype(float)
    human=np.empty((len(t),22,3))
    for i,x in enumerate(t):
        idx=int(np.searchsorted(st,x,side='right'))
        if idx==0 or idx==len(st):
            if abs(x-st[-1])<1e-10:human[i]=h[-1];continue
            raise ValueError('outside interpolation range')
        a=(x-st[idx-1])/(st[idx]-st[idx-1]);human[i]=(1-a)*h[idx-1]+a*h[idx]
    worst=0.;n=0
    with np.load(HERE/'series.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(t,saved['times'])
        for label,p in [('human',human),('robot',rob)]:
            toe=[((float(p[0,10,k])-float(p[0,7,k]))+(float(p[0,11,k])-float(p[0,8,k])))/2 for k in range(2)]
            toe=[v/norm(toe) for v in toe]
            initial=math.atan2(float(p[0,2,1]-p[0,1,1]),float(p[0,2,0]-p[0,1,0]));previous=None;unwrapped=[]
            for i in range(len(t)):
                hip=math.atan2(float(p[i,2,1]-p[i,1,1]),float(p[i,2,0]-p[i,1,0]))
                if previous is not None:
                    while hip-previous>math.pi:hip-=2*math.pi
                    while hip-previous< -math.pi:hip+=2*math.pi
                previous=hip;unwrapped.append(math.degrees(hip-initial))
                for side,s,e,w in [('left',16,18,20),('right',17,19,21)]:
                    upper=sub(p[i,s],p[i,e]);fore=sub(p[i,w],p[i,e]);relative=sub(p[i,w],p[i,s])
                    cosine=sum(a*b for a,b in zip(upper,fore))/(norm(upper)*norm(fore))
                    calculated={side+'_elbow_angle_deg':math.degrees(math.acos(max(-1,min(1,cosine)))),
                        side+'_wrist_height_relative_shoulder_m':relative[2],side+'_reach_m':norm(relative),
                        side+'_wrist_initial_toe_axis_m':sum(relative[k]*toe[k] for k in range(2)),
                        side+'_wrist_height_relative_neck_m':float(p[i,w,2])-float(p[i,12,2])}
                    for key,value in calculated.items():
                        err=abs(value-float(saved[label+'_'+key][i]));worst=max(worst,err);n+=1
            for i,value in enumerate(unwrapped):
                worst=max(worst,abs(value-float(saved[label+'_hip_yaw_change_deg'][i])));n+=1
    assert worst<1e-9
    out={'report_sha256':sha(HERE/'report.json'),'series_sha256':sha(HERE/'series.npz'),
         'frames':len(t),'values_checked':n,'max_abs_difference':worst,'training_ready':False,
         'limitations':'Same source/candidate points; no independent human reconstruction, physical contact or second semantic reviewer.'}
    with (HERE/'audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()

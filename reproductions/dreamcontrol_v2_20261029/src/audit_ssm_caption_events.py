"""Scalar recomputation of caption-review kinematic measurements and receipts."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
SRC=ROOT/'ssm_rawfps_human_batch_20261031'
CAN=ROOT/'ssm_anchored_20hz_candidates_20261031'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    review=json.loads((HERE/'report.json').read_text())
    output=json.loads((HERE/'event_measurements.json').read_text())
    assert sha(HERE/'report.json')==output['review_report_sha256']
    verified=[]
    for item,metrics in zip(review['cases'],output['cases']):
        sid=item['id']; assert sid==metrics['id']
        candidate=CAN/(sid+'.npz'); source=SRC/sid/'human_rawfps.npz'
        assert sha(candidate)==item['input_sha256'] and sha(source)==item['source_sha256']
        for page in item['pages']: assert sha(HERE/page['file'])==page['sha256']
        with np.load(candidate,allow_pickle=False) as d: t=d['sample_times'].copy(); bot=d['points_z_up_world'].copy()
        with np.load(source,allow_pickle=False) as d: st=d['source_times'].copy(); raw=d['human'].astype(np.float64)
        assert len(t)==item['frames'] and t[0]>=st[0] and t[-1]<=st[-1]
        for kind in ('human','robot'):
            values={'left':[], 'right':[]}
            for i,time in enumerate(t):
                if kind=='human':
                    upper=min(int(np.searchsorted(st,time,side='right')),len(st)-1); low=max(upper-1,0)
                    w=float((time-st[low])/(st[upper]-st[low]))
                    pose=raw[low]*(1-w)+raw[upper]*w
                else: pose=bot[i]
                for side,h,k,a,s,wrist in [('left',1,4,7,16,20),('right',2,5,8,17,21)]:
                    aa=[float(pose[h,j]-pose[k,j]) for j in range(3)]
                    bb=[float(pose[a,j]-pose[k,j]) for j in range(3)]
                    dot=sum(x*y for x,y in zip(aa,bb)); den=math.sqrt(sum(x*x for x in aa)*sum(y*y for y in bb))
                    values[side].append((math.degrees(math.acos(max(-1,min(1,dot/den)))),float(pose[wrist,2]-pose[15,2]),float(pose[wrist,2]-pose[s,2])))
            for side,rows in values.items():
                m=metrics[kind][side]; knees=[r[0] for r in rows]; wh=[r[1] for r in rows]; ws=[r[2] for r in rows]
                assert abs(min(knees)-m['min_knee_angle_deg'])<1e-8 and knees.index(min(knees))==m['min_knee_frame']
                assert abs(max(wh)-m['max_wrist_above_head_m'])<1e-10 and wh.index(max(wh))==m['max_wrist_above_head_frame']
                assert sum(v>0 for v in wh)==m['wrist_above_head_frames'] and sum(v>0 for v in ws)==m['wrist_above_shoulder_frames']
        verified.append({'id':sid,'sample_frames':len(t),'pages_hashed':len(item['pages'])})
    out={'cases':verified,'event_report_sha256':sha(HERE/'event_measurements.json'),'script_sha256':sha(Path(__file__)),
         'training_ready':False,'limitations':'Scalar recomputation uses the same source and robot arrays; images not independently redrawn; no semantic or physical acceptance.'}
    with (HERE/'event_audit.json').open('x') as f: json.dump(out,f,indent=2); f.write('\n')
    print(json.dumps(out))


if __name__=='__main__':main()

"""Compute descriptive pose-event measurements, never automatic text acceptance."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
SRC=ROOT/'ssm_rawfps_human_batch_20261031'
CAN=ROOT/'ssm_anchored_20hz_candidates_20261031'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def measurements(p):
    result={}
    for side,hip,knee,ankle,shoulder,wrist in [('left',1,4,7,16,20),('right',2,5,8,17,21)]:
        a=p[:,hip]-p[:,knee]; b=p[:,ankle]-p[:,knee]
        cosine=np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
        angles=np.degrees(np.arccos(np.clip(cosine,-1,1)))
        wrist_head=p[:,wrist,2]-p[:,15,2]
        wrist_shoulder=p[:,wrist,2]-p[:,shoulder,2]
        result[side]={'min_knee_angle_deg':float(angles.min()),'min_knee_frame':int(angles.argmin()),
                      'max_wrist_above_head_m':float(wrist_head.max()),'max_wrist_above_head_frame':int(wrist_head.argmax()),
                      'wrist_above_head_frames':int((wrist_head>0).sum()),
                      'wrist_above_shoulder_frames':int((wrist_shoulder>0).sum())}
    return result


def main():
    cases=[]
    review=json.loads((HERE/'report.json').read_text())
    for r in review['cases']:
        sid=r['id']; candidate=CAN/(sid+'.npz'); source=SRC/sid/'human_rawfps.npz'
        assert sha(candidate)==r['input_sha256'] and sha(source)==r['source_sha256']
        with np.load(source,allow_pickle=False) as d: st=d['source_times'].copy(); h=d['human'].copy()
        with np.load(candidate,allow_pickle=False) as d: t=d['sample_times'].copy(); robot=d['points_z_up_world'].copy()
        assert t[0]>=st[0] and t[-1]<=st[-1]
        human=np.column_stack([np.interp(t,st,h.reshape(len(st),66)[:,j]) for j in range(66)]).reshape(-1,22,3)
        cases.append({'id':sid,'human':measurements(human),'robot':measurements(robot),'caption_accepted':False})
    out={'cases':cases,'script_sha256':sha(Path(__file__)),'review_report_sha256':sha(HERE/'report.json'),
         'training_ready':False,'limitations':'Kinematic descriptive measurements only, points and body proportions are approximate; no physical or semantic acceptance.'}
    with (HERE/'event_measurements.json').open('x') as f: json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))


if __name__=='__main__':main()

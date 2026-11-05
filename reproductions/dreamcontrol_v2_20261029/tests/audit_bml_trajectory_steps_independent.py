"""Recheck adjacent trajectory differences by explicit frame iteration."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/bml_global_candidates_20261031')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    rows=[]
    for label in ('S04_Expert','S08_Novice'):
        folder=ROOT/label; record=json.loads((folder/'audit_report_v2.json').read_text())
        path=folder/'trajectory.npz'; assert sha(path)==record['trajectory_sha256']
        with np.load(path,allow_pickle=False) as f:
            root=f['root_wxyz_xyz'].copy(); joints=f['joint_pos'].copy(); names=[str(x) for x in f['joint_names']]
        positions=[]; joint_steps=[]
        for t in range(1,len(root)):
            positions.append(sum((float(root[t,k])-float(root[t-1,k]))**2 for k in (4,5,6))**.5)
            for j in range(27):joint_steps.append((abs(float(joints[t,j])-float(joints[t-1,j])),t-1,names[j]))
        worst=max(joint_steps)
        assert abs(max(positions)-record['max_adjacent_root_m'])<1e-7
        assert abs(worst[0]-record['max_adjacent_joint_rad'])<1e-6
        rows.append({'label':label,'audit_sha256':sha(folder/'audit_report_v2.json'),
                     'trajectory_sha256':sha(path),'max_root_step_m':max(positions),
                     'max_joint_step_rad':worst[0], 'max_joint_step_start_row':worst[1], 'joint':worst[2]})
    result={'status':'passed','records':rows,'training_ready':False,
            'limitations':'Recomputes saved trajectory steps only; no ground truth, physical or training-quality validation.'}
    with (ROOT/'adjacent_steps_independent.json').open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

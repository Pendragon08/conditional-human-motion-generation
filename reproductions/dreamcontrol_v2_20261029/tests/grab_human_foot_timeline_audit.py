"""Inspect a one-second raw GRAB human-foot timeline; no ground truth labels."""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    pilot=ROOT/'grab_human_robot_pilot_20261030';p=pilot/'pilot_arrays.npz'
    receipt=json.loads((pilot/'report.json').read_text())
    if sha(p)!=receipt['artifact_sha256']:raise ValueError('input hash mismatch')
    with np.load(p,allow_pickle=False) as d:
        joints=d['human_joints'].copy();frames=d['input_frame_indices'].copy();fps=float(d['input_fps'])
    if joints.shape!=(120,22,3) or fps!=120 or not np.all(np.diff(frames)==1):raise ValueError('unexpected timeline')
    foot=joints[:,[10,11],:];speed=np.linalg.norm(np.diff(foot,axis=0),axis=-1)*fps
    horizontal=np.linalg.norm(np.diff(foot,axis=0)[...,:2],axis=-1)*fps
    z=foot[...,2];low=speed<.05
    doc={'source_sha256':sha(p),'script_sha256':sha(__file__),'frames':len(frames),
         'first_frame':int(frames[0]),'last_frame':int(frames[-1]),'fps':fps,
         'per_foot':[{'name':name,'z_min_m':float(z[:,i].min()),'z_max_m':float(z[:,i].max()),
          'z_median_m':float(np.median(z[:,i])),'low_speed_fraction':float(low[:,i].mean()),
          'horizontal_speed_median_mps':float(np.median(horizontal[:,i])),
          'horizontal_speed_max_mps':float(horizontal[:,i].max())} for i,name in enumerate(('left','right'))],
         'warning':'One second, one S2 action; SMPL-X foot joint is not sole, z=0 is not established actual floor, and speed threshold is a proxy rather than measured contact. No robot or dynamics validation.'}
    with Path(__file__).with_name('timeline_report.json').open('x',encoding='utf8') as f:json.dump(doc,f,indent=2,allow_nan=False)
    print(json.dumps(doc),flush=True)

if __name__=='__main__':main()

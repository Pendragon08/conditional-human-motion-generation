"""Recheck GRAB S3 source-frame associations and object-origin geometry."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HUMAN=ROOT/'grab_s3_human_source_20261030';BRIDGE=ROOT/'grab_robot_feature_bridge_20261031'
ARCHIVE=ROOT/'data/GRAB/grab__s3.zip';HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HUMAN))
from grab_restricted_reader import read_grab_field
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    report=json.loads((HERE/'report.json').read_text());meta=json.loads((HUMAN/'report.json').read_text())
    assert sha(ARCHIVE)==report['archive_sha256']==meta['archive_sha256']
    assert sha(HUMAN/'human_source.npz')==report['human_sha256']
    assert sha(BRIDGE/'s3_features.npz')==report['bridge_sha256']
    for fig in report['figures']:assert sha(HERE/fig['path'])==fig['sha256']
    with np.load(HUMAN/'human_source.npz',allow_pickle=False) as f:
        h=f['human_joints'].astype(float);ids=f['input_frame_indices'].copy()
    with np.load(BRIDGE/'s3_features.npz',allow_pickle=False) as f:
        frames=f['source_frames'].copy();pts=f['points_z_up_world'].astype(float);features=f['features']
    np.testing.assert_array_equal(frames,report['source_frames'])
    assert features.shape==(len(frames)-1,263) and pts.shape==(len(frames),22,3)
    with zipfile.ZipFile(ARCHIVE) as z:obj=read_grab_field(z,meta['source_member'],'object')
    origins=np.asarray(obj['params']['transl'],dtype=float)[frames]
    aligned=h[[int(np.nonzero(ids==x)[0].item()) for x in frames]]
    # Object origin vs human head joint: not a surface, contact, mouth or bite.
    dist=[sum((float(origins[i,k])-float(aligned[i,15,k]))**2 for k in range(3))**.5 for i in range(len(frames))]
    out={'report_sha256':sha(HERE/'report.json'),'source_member':meta['source_member'],
         'frames_verified':len(frames),'first_last_frames':[int(frames[0]),int(frames[-1])],
         'apple_origin_to_human_head_joint_first_last_m':[dist[0],dist[-1]],
         'caption_accepted':False,'training_ready':False,
         'limitations':['Checks file hashes and frame mapping, not independently rerendered figures.',
                        'Object origin/head-joint distance is not mouth, object surface, contact, or evidence of eating.',
                        'Underlying FPS, model, temporal caption and physical feasibility remain unaccepted.']}
    with (HERE/'audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()

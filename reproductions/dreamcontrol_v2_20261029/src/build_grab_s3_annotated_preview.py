"""Package one manually captioned GRAB window for diagnostics only."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
ANNOTATION=ROOT/'grab_s3_manual_caption_20261101';BRIDGE=ROOT/'grab_robot_feature_bridge_20261031'
BLEND=ROOT/'pyroki_window_blend_v2_20261030/s3_blended.npz';HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    annotation=json.loads((ANNOTATION/'annotation.json').read_text());review=json.loads((ANNOTATION/'audit.json').read_text())
    bridge=json.loads((BRIDGE/'s3_report.json').read_text());bridge_audit=json.loads((BRIDGE/'s3_audit.json').read_text())
    source=BRIDGE/'s3_features.npz'
    if (sha(ANNOTATION/'annotation.json')!=review['annotation_sha256'] or sha(source)!=annotation['feature_sha256'] or
        sha(source)!=bridge['output_sha256'] or sha(BLEND)!=bridge['input_sha256'] or
        sha(BRIDGE/'s3_report.json')!=bridge_audit['report_sha256']):raise ValueError('provenance mismatch')
    if annotation['caption_origin']!='manual visual annotation by Codex; not provided by GRAB or Motion-X':
        raise ValueError('manual origin is not explicit')
    with np.load(BLEND,allow_pickle=False) as f:
        frames=f['frames20'].copy();q=f['qpos20'].astype(np.float64);names=f['joint_names'].copy()
    with np.load(source,allow_pickle=False) as f:
        features=f['features'].copy();np.testing.assert_array_equal(frames,f['source_frames'])
    if frames.tolist()!=annotation['source_frames'] or q.shape!=(20,34) or features.shape!=(19,263):raise ValueError('inconsistent frame contract')
    with source.open('rb') as src,(HERE/'features.npz').open('xb') as dst:shutil.copyfileobj(src,dst)
    if sha(HERE/'features.npz')!=annotation['feature_sha256']:raise RuntimeError('feature copy mismatch')
    with (HERE/'state.npz').open('xb') as f:
        np.savez_compressed(f,root_wxyz_xyz=q[:,:7],joint_pos=q[:,7:],sample_times=frames.astype(np.float64)/120.,joint_names=names)
    item={'sample_id':annotation['sample_id'],'source_id':'GRAB:s3/apple_eat_1.npz',
          'group_id':'conservative-family:GRAB','split':'unassigned','caption':annotation['caption'],
          'caption_source':'manual visual annotation by Codex; not GRAB or Motion-X',
          'manual_caption_receipt_sha256':sha(ANNOTATION/'annotation.json'),
          'feature_path':'features.npz','feature_sha256':sha(HERE/'features.npz'),
          'state_path':'state.npz','state_sha256':sha(HERE/'state.npz'),
          'foot_velocity_squared_threshold':0.002,'fps':20,'training_ready':False}
    out={'records':[item],'parent_blend_sha256':sha(BLEND),'bridge_report_sha256':sha(BRIDGE/'s3_report.json'),
         'annotation_audit_sha256':sha(ANNOTATION/'audit.json'),'script_sha256':sha(Path(__file__)),
         'training_ready':False,'limitations':['Unassigned single-window preview; no final training set or normalization statistics.',
         'Manual caption, source FPS, coordinate and retargeting remain bounded exploratory evidence.',
         'No real contact, collision, dynamics or complete source/license acceptance.']}
    with (HERE/'manifest.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()

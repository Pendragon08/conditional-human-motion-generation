"""Render existing 20 Hz GRAB S3 bridge with source apple-origin trace.

The filename describes the entire recording; it is not automatically a
caption for this short window. This performs no training or label acceptance.
"""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BRIDGE=ROOT/'grab_robot_feature_bridge_20261031'
HUMAN=ROOT/'grab_s3_human_source_20261030'
ARCHIVE=ROOT/'data/GRAB/grab__s3.zip'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HUMAN))
from grab_restricted_reader import read_grab_field

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    bridge=json.loads((BRIDGE/'s3_report.json').read_text());meta=json.loads((HUMAN/'report.json').read_text())
    feat=BRIDGE/'s3_features.npz';human_path=HUMAN/'human_source.npz'
    if sha(feat)!=bridge['output_sha256'] or sha(human_path)!=meta['artifact_sha256'] or sha(ARCHIVE)!=meta['archive_sha256']:
        raise ValueError('source hashes differ')
    with np.load(feat,allow_pickle=False) as f:
        frames=f['source_frames'].copy();robot=f['points_z_up_world'].astype(float);features=f['features'].copy()
    with np.load(human_path,allow_pickle=False) as f:
        human=f['human_joints'].astype(float);source_idx=f['input_frame_indices'].copy();fps=float(f['input_fps'])
    by_frame={int(frame):i for i,frame in enumerate(source_idx)}
    if len(frames)!=20 or features.shape!=(19,263) or any(int(x) not in by_frame for x in frames):raise ValueError('frame contract')
    human=human[[by_frame[int(x)] for x in frames]]
    with zipfile.ZipFile(ARCHIVE) as z:obj=read_grab_field(z,meta['source_member'],'object')
    apple=np.asarray(obj['params']['transl'],dtype=float)[frames]
    if apple.shape!=(20,3) or not np.isfinite(apple).all():raise ValueError('invalid object origin')
    edges=[(0,1),(1,4),(4,7),(7,10),(0,2),(2,5),(5,8),(8,11),(0,3),(3,6),(6,9),(9,12),(12,15),
           (9,13),(13,16),(16,18),(18,20),(9,14),(14,17),(17,19),(19,21)]
    figures=[]
    for dims in ((0,2),(1,2)):
        fig,axes=plt.subplots(4,5,figsize=(18,13),squeeze=False)
        for i,ax in enumerate(axes.flat):
            hp=human[i]-human[i,0];rp=robot[i]-robot[i,0];ap=apple[i]-human[i,0]
            for p,color in ((hp,'tab:blue'),(rp,'tab:red')):
                for a,b in edges:ax.plot(p[[a,b],dims[0]],p[[a,b],dims[1]],color=color,lw=1)
            ax.scatter([ap[dims[0]]],[ap[dims[1]]],s=15,c='green',marker='x')
            ax.set_aspect('equal');ax.set_xlim(-1,1);ax.set_ylim(-1.1,1.1)
            ax.set_title('source %d | %.3f s'%(frames[i],frames[i]/fps),fontsize=9)
        fig.suptitle('S3 apple_eat: human blue, G1 red; green apple origin relative to human pelvis only. Whole-record filename is weak context.')
        fig.tight_layout();path=HERE/('review_%d%d.png'%dims);fig.savefig(path,dpi=110);plt.close(fig);figures.append({'path':path.name,'sha256':sha(path)})
    out={'source_member':meta['source_member'],'archive_sha256':meta['archive_sha256'],'human_sha256':sha(human_path),
         'bridge_sha256':sha(feat),'bridge_report_sha256':sha(BRIDGE/'s3_report.json'),'script_sha256':sha(Path(__file__)),
         'source_frames':frames.tolist(),'source_declared_fps':fps,'nominal_sample_time_s':[float(frames[0]/fps),float(frames[-1]/fps)],
         'feature_rows':len(features),'figures':figures,'filename_context':'apple eat',
         'caption_accepted':False,'training_ready':False,
         'limitations':['Filename is sequence-level context, not a reviewed caption for a one-second crop.',
                        'Apple origin is relative to human pelvis only, not a reconstructed robot-object contact.',
                        'Human model and retargeting are approximate; no proof of real timing, ground or forces.']}
    with (HERE/'report.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()

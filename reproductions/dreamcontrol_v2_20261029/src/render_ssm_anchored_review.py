"""All-frame, pelvis-relative human/G1 contact sheets for manual caption review."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
IN = ROOT/'ssm_anchored_20hz_candidates_20261031'
HUMAN = ROOT/'ssm_rawfps_human_batch_20261031'
HERE = Path(__file__).resolve().parent
CHAINS = ((0,2,5,8,11),(0,1,4,7,10),(0,3,6,9,12,15),(9,14,17,19,21),(9,13,16,18,20))


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    result=[]
    for sid in ('007410','011120'):
        report=json.loads((IN/(sid+'_report.json')).read_text())
        candidate=IN/(sid+'.npz'); source=HUMAN/sid/'human_rawfps.npz'
        assert sha(candidate)==report['output_sha256']
        with np.load(candidate,allow_pickle=False) as d:
            times=d['sample_times'].copy(); robot=d['points_z_up_world'].copy()
        with np.load(source,allow_pickle=False) as d:
            st=d['source_times'].copy(); p=d['human'].copy()
        human_report=json.loads((HUMAN/sid/'human_report.json').read_text())
        assert sha(source)==human_report['artifact_sha256']
        assert times[0]>=st[0] and times[-1]<=st[-1]
        human=np.column_stack([np.interp(times,st,p.reshape(len(st),-1)[:,i]) for i in range(66)]).reshape(-1,22,3)
        human-=human[:,0:1]; robot-=robot[:,0:1]
        assert human.shape==robot.shape==(len(times),22,3)
        scale=float(max(np.abs(human).max(),np.abs(robot).max())*1.08)
        pages=[]
        for projection in ((0,2),(1,2)):
            for begin in range(0,len(times),40):
                fig,axes=plt.subplots(5,8,figsize=(20,13)); axes=axes.ravel()
                for j,ax in enumerate(axes):
                    i=begin+j
                    if i>=len(times): ax.axis('off'); continue
                    for a,color in ((human,'#1765ab'),(robot,'#d34132')):
                        for chain in CHAINS:
                            ax.plot(a[i,list(chain),projection[0]],a[i,list(chain),projection[1]],color=color,lw=1.3)
                    ax.set(xlim=(-scale,scale),ylim=(-scale,scale),aspect='equal',xticks=[],yticks=[])
                    ax.set_title(f'{i:03d}  {times[i]:.3f}s',fontsize=8)
                    ax.axhline(0,color='gray',lw=.3); ax.axvline(0,color='gray',lw=.3)
                title=f'{sid} | frames {begin}-{min(begin+39,len(times)-1)} | axes {projection} | human blue / robot red | pelvis-relative'
                fig.suptitle(title,fontsize=12); fig.tight_layout()
                target=HERE/f'{sid}_{projection[0]}{projection[1]}_{begin:03}.png'
                if target.exists(): raise FileExistsError(target)
                fig.savefig(target,dpi=100); plt.close(fig)
                pages.append({'file':target.name,'sha256':sha(target),'first':begin,'last':min(begin+39,len(times)-1)})
        result.append({'id':sid,'input_sha256':sha(candidate),'source_sha256':sha(source),'frames':len(times),
                       'all_caption_records':report['caption_records'],'pages':pages,'caption_accepted':False})
    with (HERE/'report.json').open('x') as f:
        json.dump({'cases':result,'script_sha256':sha(Path(__file__)),'training_ready':False,
                   'limitations':['Linear positional interpolation of fitted human, not independent ground truth.',
                    'Pelvis-relative 2D views hide world translation, depth, contact and dynamics.',
                    'Images are material for review, not semantic acceptance.']},f,indent=2); f.write('\n')
    print(json.dumps({'cases':len(result),'frames':sum(r['frames'] for r in result)}))


if __name__=='__main__': main()

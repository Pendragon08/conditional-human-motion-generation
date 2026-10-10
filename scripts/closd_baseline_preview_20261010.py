"""Render the first three newly executed motions, no best-case selection.

This is skeleton playback of saved HumanML3D-converted physics trajectories,
not an Isaac camera recording, real robot video or raw diffusion-only output.
The floor stays y=0; no per-sequence height adjustment is applied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import recover_from_ric
from closd.diffusion_planner.data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain

parser=argparse.ArgumentParser()
parser.add_argument('--root',required=True)
parser.add_argument('--asset-root',required=True)
parser.add_argument('--source-subdir',default='fresh_generation',choices=['fresh_generation','smoke'])
args=parser.parse_args()
root=Path(args.root)
source=root/args.source_subdir/'CLoSD.pkl'
if not source.exists():
    raise RuntimeError('No completed fresh generation; do not substitute released/example data.')
with source.open('rb') as f:data=pickle.load(f)
base=Path(args.asset_root)/'data/dataset/HumanML3D'
mean=torch.tensor(np.load(base/'Mean.npy'))
std=torch.tensor(np.load(base/'Std.npy'))
out=root/('previews' if args.source_subdir=='fresh_generation' else 'smoke_previews');out.mkdir(exist_ok=True)
items=[]
for idx in range(3):
    assert not (out/f'motion_{idx:02d}.mp4').exists()
    normalized=data['motion'][idx].detach().cpu()
    length=min(int(data['length'][idx]),normalized.shape[0])
    joints=recover_from_ric((normalized*std+mean).float(),22)[:length].numpy()
    assert np.isfinite(joints).all()
    fig=plt.figure(figsize=(7.2,6.0),dpi=100)
    ax=fig.add_subplot(111,projection='3d')
    # Display world x,z as ground plane and world y as vertical.
    center=(joints.min(axis=(0,1))+joints.max(axis=(0,1)))/2
    radius=1.65
    height=max(2.5,float(joints[:,:,1].max())+.25)
    ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[2]-radius,center[2]+radius),zlim=(-.25,height))
    ax.set_box_aspect((1,1,1));ax.view_init(elev=18,azim=-65)
    xx,yy=np.meshgrid([float(joints[:,:,0].min())-radius,float(joints[:,:,0].max())+radius],[float(joints[:,:,2].min())-radius,float(joints[:,:,2].max())+radius])
    ax.plot_surface(xx,yy,np.zeros_like(xx),color='#ccd6da',alpha=.2)
    ax.set_xlabel('x (m)');ax.set_ylabel('z (m)');ax.set_zlabel('height y (m)')
    caption=data['caption'][idx]
    import textwrap
    fig.suptitle('\n'.join(textwrap.wrap(caption,65)),fontsize=11)
    fig.text(.05,.025,'CLoSD '+args.source_subdir+' physics trajectory | skeleton playback | 20 fps',fontsize=9)
    colors=['#245d86','#239978','#8c4e9f','#ce7534','#af4251']
    lines=[ax.plot([],[],[],color=colors[i],lw=3,marker='o',markersize=3)[0] for i in range(len(t2m_kinematic_chain))]
    clock=ax.text2D(.02,.95,'',transform=ax.transAxes)
    def update(frame):
        # Follow root horizontally; preserve all world coordinates and y=0 floor.
        ax.set_xlim(joints[frame,0,0]-radius,joints[frame,0,0]+radius)
        ax.set_ylim(joints[frame,0,2]-radius,joints[frame,0,2]+radius)
        for line,chain in zip(lines,t2m_kinematic_chain):
            q=joints[frame,chain]
            line.set_data(q[:,0],q[:,2]);line.set_3d_properties(q[:,1])
        clock.set_text(f'index {idx} | frame {frame}/{length-1} | {frame/20:.2f}s')
        return lines+[clock]
    update(0)
    fig.savefig(out/f'motion_{idx:02d}.png')
    animation=FuncAnimation(fig,update,frames=length,interval=50,blit=False)
    video=out/f'motion_{idx:02d}.mp4'
    animation.save(video,writer=FFMpegWriter(fps=20,codec='libx264',extra_args=['-pix_fmt','yuv420p','-crf','21']))
    plt.close(fig)
    items.append({'index':idx,'caption':caption,'db_key':data['db_key'][idx],
        'frames':length,'file':video.name,'sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
        'source':args.source_subdir+' physical rollout, official HumanML3D reconstruction',
        'selection':'first three array entries; no quality selection'})
(out/'manifest.json').write_text(json.dumps(items,indent=2),encoding='utf-8')
print('CLOSD_PREVIEW_COMPLETE',json.dumps(items),flush=True)

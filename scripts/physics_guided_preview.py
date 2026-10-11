"""First zero-offset source in each family, never best-of-results selection.

Skeleton playback of recorded simulation states, not a simulator camera or
robot hardware video. A common horizontal origin is used; floor z stays zero.
"""
import argparse
import json
from pathlib import Path
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

p=argparse.ArgumentParser(); p.add_argument('--output',required=True); a=p.parse_args(); out=Path(a.output)
arms=json.loads((out/'arms.json').read_text())
names=['raw','ordinary','random_best','proposed']
packages={name:(json.loads((Path(arms[name+':11'])/'metadata.json').read_text()),
               np.load(Path(arms[name+':11'])/'trajectories.npz',allow_pickle=False)) for name in names}
base=packages['raw'][0]; body_names=base['names']
chains=[['Pelvis','L_Hip','L_Knee','L_Ankle','L_Toe'],['Pelvis','R_Hip','R_Knee','R_Ankle','R_Toe'],
        ['Pelvis','Torso','Spine','Chest','Neck','Head'],
        ['Chest','L_Thorax','L_Shoulder','L_Elbow','L_Wrist','L_Hand'],
        ['Chest','R_Thorax','R_Shoulder','R_Elbow','R_Wrist','R_Hand']]
chains=[[body_names.index(j) for j in chain] for chain in chains]
dest=out/'previews'; dest.mkdir(exist_ok=True); manifest=[]
for family in ('jump','dance'):
    i=next(i for i,c in enumerate(base['cases']) if c['family']==family and c['offset_m']==0)
    origin=packages['raw'][1]['actual'][0,i,0].copy(); origin[2]=0
    positions={name:arrays['actual'][:,i].copy()-origin for name,(_,arrays) in packages.items()}
    lengths={name:int(arrays['valid'][:,i].sum()) for name,(_,arrays) in packages.items()}
    for name,q in positions.items(): q[lengths[name]:]=q[lengths[name]-1]
    all_pos=np.concatenate(list(positions.values())); center=(all_pos.min((0,1))+all_pos.max((0,1)))/2
    radius=max(1.5,float(np.ptp(all_pos[:,:,:2],axis=(0,1)).max()/2+.4))
    goal=np.asarray(base['original_goal'][i])-origin
    fig=plt.figure(figsize=(15,5),dpi=100); lines={}; axes={}; stamps={}
    for j,name in enumerate(names):
        ax=fig.add_subplot(1,4,j+1,projection='3d'); axes[name]=ax
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),
               zlim=(-.1,max(2.4,float(all_pos[:,:,2].max()+.2))))
        ax.view_init(elev=20,azim=-60); ax.set_box_aspect((1,1,1)); ax.set_title(name)
        ax.set_xlabel('x(m)'); ax.set_ylabel('y(m)'); ax.set_zlabel('z(m)')
        xx,yy=np.meshgrid([center[0]-radius,center[0]+radius],[center[1]-radius,center[1]+radius])
        ax.plot_surface(xx,yy,np.zeros_like(xx),color='#cad6df',alpha=.2)
        ax.scatter(*goal,color='#c58300',s=45,marker='x',label='immutable goal')
        lines[name]=[ax.plot([],[],[],lw=2,marker='o',markersize=2,color=color)[0]
                     for color in ['#14618c','#199483','#69569e','#c47a36','#b74865']]
        stamps[name]=ax.text2D(.02,.95,'',transform=ax.transAxes,fontsize=9)
    caption=base['cases'][i]['caption']; fig.suptitle('\n'.join(textwrap.wrap(caption,110)),fontsize=12)
    fig.text(.03,.02,'Saved simulation skeletons | first source per family, zero offset, seed11 | common horizontal origin, fixed z=0 floor',fontsize=9)
    def update(frame):
        for name in names:
            q=positions[name][frame]
            for line,chain in zip(lines[name],chains):
                line.set_data(q[chain,0],q[chain,1]); line.set_3d_properties(q[chain,2])
            status='TERMINATED' if frame>=lengths[name] else 'running'
            stamps[name].set_text('%.2fs | %s'%(frame/30,status))
        return [line for group in lines.values() for line in group]
    update(90); fig.savefig(dest/(family+'_comparison.png'))
    animation=FuncAnimation(fig,update,frames=range(0,120,3),interval=100,blit=False)
    animation.save(dest/(family+'_comparison.mp4'),writer=FFMpegWriter(fps=10,codec='libx264',
                   extra_args=['-pix_fmt','yuv420p','-crf','24']))
    plt.close(fig)
    manifest.append(dict(family=family,case_id=base['cases'][i]['case_id'],seed=11,lengths=lengths,
                         selection='first source per family, zero offset, independent of quality',
                         video=family+'_comparison.mp4',image=family+'_comparison.png'))
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('PHYSICS_GUIDED_PREVIEW_COMPLETE')

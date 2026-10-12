"""CPU skeleton visualization of saved seed11 replays, not a robot video.

First zero-offset source per family, chosen independently of success scores.
No generation, simulation or best-looking example selection.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/paired_feedback_validation_v51_20261012'
DEST = OUT/'previews'
assert not DEST.exists(), 'Existing preview attempt retained; inspect before any new invocation'
packages = {}
for arm in ('feedback','random'):
    path = OUT/'raw'/('val_%s_replay_s11' % arm)
    meta = json.loads((path/'metadata.json').read_text())
    with np.load(path/'trajectories.npz',allow_pickle=False) as archive:
        arrays = {name:archive[name] for name in ('actual','reference','valid')}
    packages[arm] = (meta, arrays, path)
base = packages['feedback'][0]
assert base['original_goal_sha256'] == packages['random'][0]['original_goal_sha256']
assert base['cases'] == packages['random'][0]['cases']
chains = [['Pelvis','L_Hip','L_Knee','L_Ankle','L_Toe'],
          ['Pelvis','R_Hip','R_Knee','R_Ankle','R_Toe'],
          ['Pelvis','Torso','Spine','Chest','Neck','Head'],
          ['Chest','L_Thorax','L_Shoulder','L_Elbow','L_Wrist','L_Hand'],
          ['Chest','R_Thorax','R_Shoulder','R_Elbow','R_Wrist','R_Hand']]
chains = [[base['names'].index(name) for name in chain] for chain in chains]
DEST.mkdir()
manifest = []
for family in ('jump','dance'):
    index = next(i for i,c in enumerate(base['cases']) if c['family']==family and c['offset_m']==0)
    origin = packages['feedback'][1]['actual'][0,index,0].copy()
    origin[2] = 0
    lengths = {arm:int(data['valid'][:,index].sum()) for arm,(_,data,_) in packages.items()}
    assert min(lengths.values()) > 0
    positions = {}
    for arm,(_,data,_) in packages.items():
        positions[arm] = {kind:data[kind][:,index].copy()-origin for kind in ('actual','reference')}
        for pose in positions[arm].values():
            pose[lengths[arm]:] = pose[lengths[arm]-1]
    combined = np.concatenate([p for x in positions.values() for p in x.values()])
    center = (combined.min((0,1))+combined.max((0,1)))/2
    radius = max(1.5,float(np.ptp(combined[:,:,:2],axis=(0,1)).max()/2+.4))
    goal = np.asarray(base['original_goal'][index])-origin
    fig = plt.figure(figsize=(10,5),dpi=100)
    lines = {}; stamps = {}
    for j,arm in enumerate(packages):
        ax=fig.add_subplot(1,2,j+1,projection='3d')
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),
               zlim=(-.1,max(2.4,float(combined[:,:,2].max()+.2))))
        ax.view_init(elev=20,azim=-60); ax.set_box_aspect((1,1,1))
        ax.set_title(arm+' | solid: executed; grey: reference')
        ax.set_xlabel('x(m)');ax.set_ylabel('y(m)');ax.set_zlabel('z(m)')
        xx,yy=np.meshgrid([center[0]-radius,center[0]+radius],[center[1]-radius,center[1]+radius])
        ax.plot_surface(xx,yy,np.zeros_like(xx),color='#cad6df',alpha=.2)
        ax.scatter(*goal,color='#b67500',s=45,marker='x')
        lines[arm]={kind:[ax.plot([],[],[],lw=2 if kind=='actual' else 1,
                    color=('#14618c' if kind=='actual' else '#999999'),alpha=1 if kind=='actual' else .5)[0]
                    for _ in chains] for kind in ('actual','reference')}
        stamps[arm]=ax.text2D(.02,.92,'',transform=ax.transAxes,fontsize=9)
    fig.suptitle(family+' | seed11; first source; zero target offset',fontsize=12)
    fig.text(.03,.02,'Saved human simulation skeletons; NOT simulator-camera or real-robot footage. Immutable goal: gold cross.',fontsize=8)
    def update(frame):
        for arm in packages:
            for kind in lines[arm]:
                q=positions[arm][kind][frame]
                for line,chain in zip(lines[arm][kind],chains):
                    line.set_data(q[chain,0],q[chain,1]);line.set_3d_properties(q[chain,2])
            ended=frame>=lengths[arm]
            stamps[arm].set_text('recorded step %d | %.2fs | %s'%(frame,(frame+1)*base['dt'],
                'TERMINATED: held last state' if ended else 'recorded actual state'))
        return []
    update(90);fig.savefig(DEST/(family+'_paired.png'))
    movie=FuncAnimation(fig,update,frames=range(0,120,3),interval=100,blit=False)
    movie.save(DEST/(family+'_paired.mp4'),writer=FFMpegWriter(fps=10,codec='libx264',
                 extra_args=['-pix_fmt','yuv420p','-crf','24']))
    plt.close(fig)
    manifest.append(dict(family=family,case=base['cases'][index],lengths=lengths,
        seed=11,selection='First zero-offset source per family, independent of quality',
        evaluation_type='Saved simulation-state skeleton visualization, no new physics or robot execution',
        trajectory_hashes={arm:hashlib.sha256((path/'trajectories.npz').read_bytes()).hexdigest()
                           for arm,(_,_,path) in packages.items()},
        video=family+'_paired.mp4',image=family+'_paired.png'))
(DEST/'manifest.json').write_text(json.dumps(dict(examples=manifest,simulation_calls=0,generation_calls=0,
    confirmed_innovation=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
print('PAIRED_FEEDBACK_SKELETON_PREVIEWS_COMPLETE',flush=True)

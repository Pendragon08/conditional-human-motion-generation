"""Visualize fixed original baseline human samples; no generation, fitting or physics."""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
DATA=ROOT/'artifacts/human_motion_report_20261002'
os.environ['MPLCONFIGDIR']=str(DATA/'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import json, hashlib, subprocess

OUT=ROOT/'media/original_human_generation_20261002';OUT.mkdir(exist_ok=True)
CHAIN=[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]
manifest=[]
for source in json.loads((DATA/'source_manifest.json').read_text()):
    arrays=np.load(DATA/source['output'],allow_pickle=False)
    for i,ref in enumerate(source['selection_ids']):
        length=int(arrays['lengths'][i]);motion=arrays['motion'][i,:,:,:length].transpose(2,0,1)
        hints=arrays['hint'][i,:length];assert motion.shape==(196,22,3) and np.isfinite(motion).all()
        # Only display-axis permutation (x,y-up,z-forward)->(x,z,y-up).
        xyz=motion[:,:,[0,2,1]];target=hints[:,:,[0,2,1]];mask=np.any(hints!=0,axis=-1)
        bounds=np.concatenate([xyz.reshape(-1,3),target[mask]])
        lo=bounds.min(0);hi=bounds.max(0);mid=(lo+hi)/2
        width=max(2.0,float((hi-lo).max())+.35)
        fig=plt.figure(figsize=(12.8,7.2),dpi=100,facecolor='#f0f5f9')
        axes=[fig.add_subplot(121,projection='3d'),fig.add_subplot(122,projection='3d')]
        handles=[]
        for ax,azim,title in zip(axes,[-60,0],['Oblique view','Side view']):
            ax.set_proj_type('ortho');ax.view_init(elev=16 if azim==-60 else 0,azim=azim)
            ax.set_xlim(mid[0]-width/2,mid[0]+width/2);ax.set_ylim(mid[1]-width/2,mid[1]+width/2)
            ax.set_zlim(-.12,width-.12);ax.set_box_aspect([1,1,1]);ax.set_facecolor('#f0f5f9')
            ax.set_xlabel('lateral (m)',labelpad=4);ax.set_ylabel('forward (m)',labelpad=4);ax.set_zlabel('height (m)',labelpad=4)
            ax.set_title(title,fontsize=13,color='#24435b');ax.grid(alpha=.25)
            gx=np.linspace(mid[0]-width/2,mid[0]+width/2,6);gy=np.linspace(mid[1]-width/2,mid[1]+width/2,6)
            for x in gx:ax.plot([x,x],[gy[0],gy[-1]],[0,0],color='#cfdae1',linewidth=.6)
            for y in gy:ax.plot([gx[0],gx[-1]],[y,y],[0,0],color='#cfdae1',linewidth=.6)
            bones=[ax.plot([],[],[],linewidth=5 if n!=2 else 7,color='#df8733' if n==0 else '#217d9b',solid_capstyle='round')[0] for n in range(5)]
            head=ax.plot([],[],[],marker='o',markersize=11,color='#217d9b')[0]
            goal=ax.plot([],[],[],marker='x',markersize=13,markeredgewidth=2.5,linestyle='None',color='#c84f53')[0]
            rt=ax.plot([],[],[],color='#df8733',linewidth=1.1,alpha=.7)[0]
            lt=ax.plot([],[],[],color='#579e8d',linewidth=1.1,alpha=.65)[0]
            handles.append((bones,head,goal,rt,lt))
        fig.suptitle(f'Original OmniControl human motion | Kick | seed {source["seed"]} | sample {ref:03d}',fontsize=17,color='#16364d',y=.96)
        fig.text(.5,.9,'Text: A person stands and kicks with his right leg.',ha='center',fontsize=12,color='#4b6578')
        time=fig.text(.5,.055,'',ha='center',fontsize=12,color='#16364d')
        fig.text(.5,.018,'Blue: body    Orange: right leg / foot trace    Red cross: active spatial hint    Green trace: left foot',ha='center',fontsize=10,color='#577185')
        fig.subplots_adjust(left=.01,right=.99,bottom=.1,top=.85,wspace=.02)
        p=OUT/f'Kick_seed{source["seed"]}_sample{ref:03d}.mp4'
        writer=FFMpegWriter(fps=20,codec='libx264',bitrate=1700,extra_args=['-pix_fmt','yuv420p','-threads','1'])
        with writer.saving(fig,str(p),100):
            for f in range(length):
                for bones,head,goal,rt,lt in handles:
                    for bone,chain in zip(bones,CHAIN):
                        pts=xyz[f,chain];bone.set_data_3d(pts[:,0],pts[:,1],pts[:,2])
                    hp=xyz[f,15];head.set_data_3d([hp[0]],[hp[1]],[hp[2]])
                    gt=target[f,mask[f]];goal.set_data_3d(gt[:,0],gt[:,1],gt[:,2])
                    for line,j in [(rt,11),(lt,10)]:
                        pts=xyz[max(0,f-30):f+1,j];line.set_data_3d(pts[:,0],pts[:,1],pts[:,2])
                time.set_text(f'time {f/20:.2f} s / {length/20:.2f} s  |  original 20 fps, 1x speed  |  not a physics rollout')
                writer.grab_frame()
                if f==30:fig.savefig(str(p.with_suffix('.jpg')),dpi=100)
        plt.close(fig)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames:format=duration','-of','json',str(p)],text=True))
        assert int(probe['streams'][0]['nb_frames'])==length and float(probe['format']['duration'])==length/20
        manifest.append(dict(file=p.name,seed=source['seed'],sample_id=ref,text=str(arrays['text'][i]),source=source['source'],source_sha256=source['source_sha256'],source_batch_count=100,
            selection='fixed first three IDs before viewing outcomes',frames=length,fps=20,duration_seconds=length/20,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,ffprobe=probe,
            display='original 22-joint world-coordinate skeleton, two fixed views, y-up axis permutation only; no smoothing, root recentering, IK or physics',
            scope='baseline original guided diffusion generation, not innovation candidate; single Kick text class'))
        print('HUMAN_MOTION_RENDERED',p.name,flush=True)
(OUT/'manifest.json').write_text(json.dumps({'status':'COMPLETE','render_only':True,'videos':manifest,'model_source':'DreamControl TrajGen official pretrained OmniControl model_humanml3d.pt'},indent=2))
print('ORIGINAL_HUMAN_REPORT_COMPLETE',len(manifest),flush=True)

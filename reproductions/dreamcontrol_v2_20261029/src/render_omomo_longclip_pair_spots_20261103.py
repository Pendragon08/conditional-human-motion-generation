"""Compare sampled original human video and 29-DoF robot trajectories.

This is targeted review material for 15 cropped-source groups; it does not
establish actual synchronization, contact, caption truth, or source rights.
"""
import hashlib,json,shutil
from pathlib import Path
import cv2,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
U=R/'omni29_uncropped_review_20261103';V=R/'omomo_longclip_source_video_v2_20261103';O=R/'omomo_longclip_pair_spots_20261103'
CHAINS=((0,2,5,8,11),(0,1,4,7,10),(0,3,6,9,12,15),(9,14,17,19,21),(9,13,16,18,20))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if O.exists():raise FileExistsError(O)
 if shutil.disk_usage(R).free<8*1024**3+356*1024**2:raise RuntimeError('disk reserve')
 audit=json.loads((V/'audit.json').read_text());meta=json.loads((U/'report.json').read_text());inp=U/'records.jsonl'
 if sha(inp)!=audit['robot_records_sha256'] or sha(inp)!=meta['records_sha256']:raise ValueError('source binding')
 sources=[json.loads(x) for x in inp.read_text().splitlines()];bygroup={r['nominal_source_group'].split(':',1)[1]:r for r in sources if r['source_member'].endswith('_original.npz')}
 if len(bygroup)!=15 or audit['videos']!=15:raise ValueError('group count')
 O.mkdir(exist_ok=False);n=0
 with (O/'records.jsonl').open('x',encoding='utf8',buffering=1) as log:
  for a in audit['details']:
   if shutil.disk_usage(R).free<8*1024**3+356*1024**2:raise RuntimeError('disk reserve')
   group=a['group'];row=bygroup[group];video=V/(group+'.mp4');state=U/row['output']
   if sha(video)!=a['video_sha256'] or sha(state)!=row['output_sha256']:raise ValueError('source hash')
   with np.load(state,allow_pickle=False) as f:points=f['points_z_up'];q=f['qpos29_source_layout_20hz'];times=f['sample_times']
   cap=cv2.VideoCapture(str(video));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
   if frames!=a['video_frames'] or abs(fps-a['video_fps'])>1e-8:raise ValueError('video metadata')
   idx=np.rint(np.linspace(0,len(times)-1,12)).astype(int);coords=np.concatenate([points.reshape(-1,3),q[:,40:43]],axis=0);center=(coords.max(0)+coords.min(0))/2;radius=max(.7,float(np.ptp(coords,axis=0).max()*.6));pages=[]
   for page in range(2):
    fig,axes=plt.subplots(6,3,figsize=(10,16),constrained_layout=True)
    for rr,i in enumerate(idx[page*6:(page+1)*6]):
     vf=int(np.rint(times[i]*fps));cap.set(cv2.CAP_PROP_POS_FRAMES,vf);ok,frame=cap.read()
     if not ok:raise ValueError('video frame read')
     axes[rr,0].imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));axes[rr,0].set_title(f'human frame {vf} | nominal {times[i]:.2f}s',fontsize=7);axes[rr,0].axis('off')
     for col,(x,y) in enumerate(((0,2),(1,2)),start=1):
      ax=axes[rr,col]
      for chain in CHAINS:ax.plot(points[i,list(chain),x],points[i,list(chain),y],'-o',markersize=1.8,lw=1)
      ax.plot(q[i,40+x],q[i,40+y],'ks',ms=5);ax.set(xlim=(center[x]-radius,center[x]+radius),ylim=(center[y]-radius,center[y]+radius),aspect='equal',title=('robot x-z' if col==1 else 'robot y-z')+'; object origin',xticks=[],yticks=[])
    fig.suptitle(group+' | '+row['caption']+'\nHuman source, 29DoF robot and object origin; sampled instants only',fontsize=8)
    p=O/f'{group}_page{page+1}.png';fig.savefig(p,dpi=95);plt.close(fig);pages.append({'file':p.name,'sha256':sha(p)})
   cap.release();record={'group':group,'sample_id':row['sample_id'],'caption':row['caption'],'video_sha256':a['video_sha256'],
    'state_sha256':row['output_sha256'],'sample_indices':idx.tolist(),'time_seconds':[float(times[i]) for i in idx],
    'pages':pages,'caption_accepted':False}
   log.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');n+=1;print(json.dumps({'groups_processed':n,'group':group}),flush=True)
 out={'groups':n,'video_audit_sha256':sha(V/'audit.json'),'robot_records_sha256':sha(inp),'records_sha256':sha(O/'records.jsonl'),
      'training_ready':False,'limitations':['12 time samples per group, not full-frame semantic review.',
      'Robot object shown as origin, not geometry, support, or contact.',
      'Names and nominal sample times do not prove synchronization or identity.',
      'Only original retargeted reference is shown; augmented variants are not covered.']}
 with (O/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False)
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()

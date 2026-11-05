"""12 nominal-time visual samples from original human video and 29DoF G1.

Images support a limited semantic review only; projection does not show forces
or contacts and name/time agreement does not prove recording identity.
"""
import hashlib,json
from pathlib import Path
import cv2,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'omni29_direct_text_motion_pilot_20261103'
VIDEO=R/'omomo_source_video_20261102/sub3_largebox_011.mp4'
CHAINS=((0,2,5,8,11),(0,1,4,7,10),(0,3,6,9,12,15),(9,14,17,19,21),(9,13,16,18,20))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 report=json.loads((D/'report.json').read_text());data=D/'state_features.npz'
 if sha(data)!=report['output_sha256'] or sha(VIDEO)!='f118e60ef858ca8f15de81ae1fa98aa3a548740055e9c4239e10e95efeadb609':raise ValueError('evidence hash')
 with np.load(data,allow_pickle=False) as f:points=f['points_z_up'];q=f['qpos29_source_layout_20hz'];times=f['sample_times']
 cap=cv2.VideoCapture(str(VIDEO));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
 if n!=192 or abs(fps-30)>1e-6:raise ValueError('source video metadata')
 idx=np.rint(np.linspace(0,len(times)-1,12)).astype(int);allp=np.concatenate([points.reshape(-1,3),q[:,40:43]],axis=0);center=(allp.max(0)+allp.min(0))/2;radius=max(.7,float(np.ptp(allp,axis=0).max()*.6))
 out=R/'omni29_text_spot_review_20261103';out.mkdir(exist_ok=False);images=[]
 for page in range(2):
  fig,axes=plt.subplots(6,3,figsize=(10,16),constrained_layout=True)
  for row,i in enumerate(idx[page*6:(page+1)*6]):
   vf=int(np.rint(times[i]*fps));cap.set(cv2.CAP_PROP_POS_FRAMES,vf);ok,frame=cap.read()
   if not ok:raise ValueError('video frame read failed')
   axes[row,0].imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));axes[row,0].set_title(f'human video frame {vf}, t={times[i]:.2f}s',fontsize=7);axes[row,0].axis('off')
   for col,(a,b) in enumerate(((0,2),(1,2)),start=1):
    ax=axes[row,col]
    for chain in CHAINS:ax.plot(points[i,list(chain),a],points[i,list(chain),b],'-o',markersize=1.8,lw=1)
    ax.plot(q[i,40+a],q[i,40+b],'ks',ms=5);ax.set(xlim=(center[a]-radius,center[a]+radius),ylim=(center[b]-radius,center[b]+radius),aspect='equal',title=('robot x-z' if col==1 else 'robot y-z')+'; object center',xticks=[],yticks=[])
    ax.grid(alpha=.2)
  fig.suptitle(report['caption']+'\n29DoF robot points and object center; sampled frames, no box geometry',fontsize=9)
  path=out/f'page_{page+1}.png';fig.savefig(path,dpi=105);plt.close(fig);images.append({'path':path.name,'sha256':sha(path)})
 cap.release()
 result={'caption':report['caption'],'source_video_sha256':sha(VIDEO),'artifact_sha256':sha(data),'nominal_20hz_indices':idx.tolist(),'source_video_fps':fps,'images':images,'caption_accepted':False,'training_ready':False,'limitations':['12 sampled moments only; not a full sequence review.','Object shown as center, not geometry or contact.','Matching names and nominal timestamps do not prove recording identity or true synchronization.']}
 with (out/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps(result))
if __name__=='__main__':main()

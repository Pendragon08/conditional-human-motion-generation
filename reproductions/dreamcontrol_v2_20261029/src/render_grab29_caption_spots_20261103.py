"""Render limited robot/object spot checks; not a caption acceptance or physical filter."""
import hashlib,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'grab_omni_mixed29_full_preview_20261103'
S=['dex29_s1_airplane_offhand_1','dex29_s1_alarmclock_offhand_1','dex29_s1_apple_offhand_1','dex29_s9_wineglass_pass_1']
EDGES=[(0,1),(1,4),(4,7),(7,10),(0,2),(2,5),(5,8),(8,11),(0,3),(3,6),(6,9),(9,12),(12,15),(9,13),(13,16),(16,18),(18,20),(9,14),(14,17),(17,19),(19,21)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if shutil.disk_usage(R).free<8*1024**3+356*1024**2:raise RuntimeError('disk reserve')
 report=json.loads((D/'report.json').read_text());manifest=D/'manifest.json'
 if sha(manifest)!=report['manifest_sha256']:raise ValueError('manifest hash')
 rows={r['sample_id']:r for r in json.loads(manifest.read_text())['records']};out=R/'grab29_caption_spots_20261103';out.mkdir(exist_ok=False);results=[]
 for sid in S:
  r=rows[sid];path=R/r['state_path']
  if sha(path)!=r['state_sha256']:raise ValueError('state hash')
  with np.load(path,allow_pickle=False) as f:
   pts=np.asarray(f['points_z_up']);obj=np.asarray(f['object_pose_wxyz_xyz']);names=f['object_names'].tolist()
  frames=np.unique(np.linspace(0,len(pts)-1,12,dtype=int));fig,axes=plt.subplots(2,len(frames),figsize=(2*len(frames),5),squeeze=False)
  for col,i in enumerate(frames):
   p=pts[i]-pts[i,0];o=obj[i,0,4:7]-pts[i,0]
   for row,dims in enumerate(((0,2),(1,2))):
    ax=axes[row,col]
    for a,b in EDGES:ax.plot(p[[a,b],dims[0]],p[[a,b],dims[1]],color='steelblue',lw=1.2)
    ax.scatter(o[dims[0]],o[dims[1]],c='red',marker='x',s=28)
    ax.set_aspect('equal');ax.set_xlim(-1.2,1.2);ax.set_ylim(-1.2,1.2);ax.set_xticks([]);ax.set_yticks([])
    if row==0:ax.set_title(f'{i/20:.2f}s',fontsize=8)
    if col==0:ax.set_ylabel(('x-z','y-z')[row])
  fig.suptitle(sid+' | '+r['caption']+' | red x: '+str(names[0])+' origin; sampled 12 frames, nominal 20 Hz',fontsize=10)
  fig.tight_layout();dest=out/(sid+'.png');fig.savefig(dest,dpi=110);plt.close(fig)
  results.append({'sample_id':sid,'caption':r['caption'],'state_sha256':r['state_sha256'],'cropped':r['cropped_for_max_10s'],'object_names':names,'sampled_frames':frames.tolist(),'image_sha256':sha(dest),'image':dest.name})
 with (out/'report.json').open('x',encoding='utf8') as f:json.dump({'manifest_sha256':sha(manifest),'samples':results,'semantics_accepted':False,'limitations':['12 nominal-time instants only, not full temporal review.','An object center is not object shape or contact; no second person or external force is represented.','No physical-action filtering.']},f,indent=2,ensure_ascii=False)
 print(json.dumps(results))
if __name__=='__main__':main()

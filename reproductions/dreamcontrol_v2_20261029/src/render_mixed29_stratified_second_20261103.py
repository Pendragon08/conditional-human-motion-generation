"""Render one candidate per review stratum; sampled views are not label acceptance."""
import hashlib,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
Q=R/'grab_omni_caption_time_triage_20261103'
O=R/'grab_omni_stratified_second_20261103'
EDGES=[(0,1),(1,4),(4,7),(7,10),(0,2),(2,5),(5,8),(8,11),(0,3),(3,6),(6,9),(9,12),(12,15),(9,13),(13,16),(16,18),(18,20),(9,14),(14,17),(17,19),(19,21)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if O.exists():raise FileExistsError(O)
 if shutil.disk_usage(R).free<8*1024**3+356*1024**2:raise RuntimeError('disk reserve')
 report=json.loads((Q/'report.json').read_text());queue=Q/'stratified_review_queue.jsonl'
 if sha(queue)!=report['review_queue_sha256']:raise ValueError('queue hash')
 rows=[json.loads(x) for x in queue.read_text(encoding='utf8').splitlines()];groups={};selected={}
 for r in rows:groups.setdefault(r['stratum'],[]).append(r)
 for bucket,items in groups.items():selected[bucket]=items[1]
 if len(selected)!=6:raise ValueError('strata count')
 O.mkdir(exist_ok=False);items=[]
 for bucket,r in selected.items():
  path=R/r['state_path']
  if sha(path)!=r['state_sha256']:raise ValueError('state hash')
  with np.load(path,allow_pickle=False) as f:
   pts=np.asarray(f['points_z_up'])
   if 'object_pose_wxyz_xyz' in f:obj=np.asarray(f['object_pose_wxyz_xyz'])[:,0,4:7]
   else:obj=np.asarray(f['qpos29_source_layout_20hz'])[:,40:43]
  if pts.shape!=(len(obj),22,3) or not np.isfinite(obj).all():raise ValueError('object shape')
  frames=np.unique(np.linspace(0,len(pts)-1,12,dtype=int));fig,axes=plt.subplots(2,len(frames),figsize=(2*len(frames),5),squeeze=False)
  for col,i in enumerate(frames):
   p=pts[i]-pts[i,0];o=obj[i]-pts[i,0]
   for row,dims in enumerate(((0,2),(1,2))):
    ax=axes[row,col]
    for a,b in EDGES:ax.plot(p[[a,b],dims[0]],p[[a,b],dims[1]],color='steelblue',lw=1.1)
    ax.scatter(o[dims[0]],o[dims[1]],c='red',marker='x',s=28)
    ax.set_aspect('equal');ax.set_xlim(-1.5,1.5);ax.set_ylim(-1.5,1.5);ax.set_xticks([]);ax.set_yticks([])
    if row==0:ax.set_title(f'{i/20:.2f}s',fontsize=8)
    if col==0:ax.set_ylabel(('x-z','y-z')[row])
  fig.suptitle(r['sample_id']+' | '+r['caption']+' | red x = object origin; sampled 12 instants',fontsize=9)
  fig.tight_layout();dest=O/(r['sample_id'].replace('/','_')+'.png');fig.savefig(dest,dpi=110);plt.close(fig)
  items.append({'stratum':bucket,'sample_id':r['sample_id'],'caption':r['caption'],'crop_10s':r['crop_10s'],
                'triage_reasons':r['triage_reasons'],'state_sha256':r['state_sha256'],'frames':frames.tolist(),
                'image':dest.name,'image_sha256':sha(dest)})
 with (O/'report.json').open('x',encoding='utf8') as f:json.dump({'queue_sha256':sha(queue),'items':items,
   'limitations':['12 sampled instants, not complete motion review.','Object center does not establish geometry, contact or physical event semantics.','Six selected from stratified queue are not a population-quality estimate.'],
   'caption_accepted':False},f,indent=2,ensure_ascii=False,allow_nan=False)
 print(json.dumps(items,ensure_ascii=False))
if __name__=='__main__':main()

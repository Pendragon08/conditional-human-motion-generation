"""Compare two 29-DoF URDF link-origin FK on 33 synthetic configurations."""
import hashlib,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
U=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
V=R/'omni_g1_urdf_20261030/models/g1/g1_29dof.urdf'
sys.path.insert(0,str(R/'ssm_new6_20hz_resume_20261101'))
from g1_fk import URDFFK
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def limits(p):return {j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in ET.parse(p).getroot().findall('joint') if j.get('type') in ('revolute','prismatic') and j.find('limit') is not None}
def main():
 out=R/'gmr_grab_29dof_smoke_20261103/omni29_dream29_fk_audit.json'
 if out.exists():raise FileExistsError(out)
 a,b=URDFFK(U),URDFFK(V);names=list(a.actuated);links=sorted(set(a.links)&set(b.links))
 if len(names)!=29 or set(names)!=set(b.actuated):raise ValueError('different joint sets')
 al,bl=limits(U),limits(V);rng=np.random.default_rng(20261103);q=np.zeros((33,29))
 for k,name in enumerate(names):
  lo=max(al[name][0],bl[name][0],-.5);hi=min(al[name][1],bl[name][1],.5)
  if lo>=hi:raise ValueError('disjoint limits')
  q[1:,k]=rng.uniform(lo,hi,32)
 pos=np.zeros((33,3));quat=np.tile([0.,0.,0.,1.],(33,1))
 diff=np.linalg.norm(a.forward(pos,quat,q,names,links=links)-b.forward(pos,quat,q,names,links=links),axis=-1)
 i,j=np.unravel_index(np.argmax(diff),diff.shape)
 report={'dream_urdf_sha256':sha(U),'omni_urdf_sha256':sha(V),'samples':33,'joint_count':29,'links':links,'max_difference_m':float(diff.max()),'zero_difference_m':float(diff[0].max()),'worst_link':links[j],'max_by_link':{name:float(diff[:,k].max()) for k,name in enumerate(links)},'training_ready':False,'limitations':['Synthetic configurations; link origins only.','No evidence about meshes, inertias, joint limits, source semantics, physical tracking or text.']}
 with out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
 print(json.dumps({'joints':len(names),'links':len(links),'max_m':report['max_difference_m'],'zero_m':report['zero_difference_m'],'worst':report['worst_link']}))
if __name__=='__main__':main()

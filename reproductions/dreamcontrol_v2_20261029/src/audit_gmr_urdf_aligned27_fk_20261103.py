"""Compare project GMR locked model against independently parsed G1 URDF FK."""
import hashlib,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
MODEL=ROOT/'gmr_urdf_aligned27_20261103/g1_gmr_locked27_aligned.xml'
sys.path.insert(0,str(ROOT/'ssm_new6_20hz_resume_20261101'))
from g1_fk import URDFFK
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 pin={str(p):sha(p) for p in (MODEL,URDF)};m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);fk=URDFFK(URDF)
 names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(m.njnt) if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE)]
 links=sorted(set(fk.links)&{mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,j) for j in range(m.nbody) if mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,j)})
 if len(names)!=27 or set(names)!=set(fk.actuated) or len(links)<20:raise ValueError('joint/body coverage')
 limits={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in ET.parse(URDF).getroot().findall('joint') if j.get('type') in ('revolute','prismatic') and j.find('limit') is not None}
 rng=np.random.default_rng(20261103);n=33;angles=np.zeros((n,27))
 for k,name in enumerate(names):
  j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name);lo=max(limits[name][0],m.jnt_range[j,0]);hi=min(limits[name][1],m.jnt_range[j,1]);
  if lo>=hi:raise ValueError('empty joint intersection')
  angles[1:,k]=rng.uniform(max(lo,-.5),min(hi,.5),n-1)
 ref=fk.forward(np.zeros((n,3)),np.tile([0.,0.,0.,1.],(n,1)),angles,names,links=links)
 found=np.empty_like(ref)
 for i in range(n):
  mujoco.mj_resetData(m,d);d.qpos[:7]=[0,0,0,1,0,0,0]
  for k,name in enumerate(names):j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name);d.qpos[m.jnt_qposadr[j]]=angles[i,k]
  mujoco.mj_forward(m,d)
  found[i]=[d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)] for name in links]
 dif=np.linalg.norm(found-ref,axis=-1);i,j=np.unravel_index(np.argmax(dif),dif.shape)
 if any(sha(Path(p))!=v for p,v in pin.items()):raise RuntimeError('model changed')
 report={'model_sha256':pin[str(MODEL)],'urdf_sha256':pin[str(URDF)],'frames':n,'links':links,'max_difference_m':float(dif.max()),'mean_difference_m':float(dif.mean()),'worst_frame':int(i),'worst_link':links[j], 'zero_pose_max_difference_m':float(dif[0].max()),'training_ready':False,'limitations':['Synthetic joint states only; free roots fixed to identity.','Compares link origins, not collision meshes, joint torques, data semantics or dynamics.']}
 with (ROOT/'gmr_urdf_aligned27_20261103/fk_audit.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
 print(json.dumps({'frames':n,'links':len(links),'max_m':report['max_difference_m'],'mean_m':report['mean_difference_m'],'zero_m':report['zero_pose_max_difference_m'],'worst_link':report['worst_link']}))
if __name__=='__main__':main()

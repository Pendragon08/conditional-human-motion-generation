"""Create a project-local 27-DoF GMR model by locking two waist joints.

This modifies a copy in a new directory; the upstream GMR assets are read-only.
It is an explicit GMR-based adaptation, not the author's DreamControl-v2 model.
"""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
GMR=Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/GMR')
SOURCE=GMR/'assets/unitree_g1/g1_mocap_29dof.xml'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
CONFIG=GMR/'general_motion_retargeting/ik_configs/smplx_to_g1.json'
OUT=ROOT/'gmr_locked27_model_20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 pins={str(p):sha(p) for p in (SOURCE,URDF,CONFIG)};tree=ET.parse(SOURCE);root=tree.getroot()
 compiler=root.find('compiler');compiler.set('meshdir',str((SOURCE.parent/'meshes').resolve(strict=True)))
 removed=[]
 for name in ('waist_roll_joint','waist_pitch_joint'):
  matches=[(parent,child) for parent in root.iter() for child in list(parent) if child.tag=='joint' and child.get('name')==name]
  if len(matches)!=1:raise ValueError('unexpected waist joint definition: '+name)
  matches[0][0].remove(matches[0][1]); motors=[(parent,child) for parent in root.iter() for child in list(parent) if child.tag=='motor' and child.get('joint')==name]
  if len(motors)!=1:raise ValueError('unexpected waist actuator definition: '+name)
  motors[0][0].remove(motors[0][1]);removed.append(name)
 OUT.mkdir(exist_ok=False);target=OUT/'g1_gmr_locked27.xml'
 with target.open('xb') as f:tree.write(f,encoding='utf-8',xml_declaration=True)
 model=mujoco.MjModel.from_xml_path(str(target))
 names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(model.njnt) if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE)]
 limits={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in ET.parse(URDF).getroot().findall('joint') if j.get('type') in ('revolute','prismatic') and j.find('limit') is not None}
 if model.nq!=34 or model.nv!=33 or model.nu!=27 or len(names)!=27 or set(names)!=set(limits):raise ValueError('model joint set or dimensionality mismatch')
 config=json.loads(CONFIG.read_text());bodies=set(config['ik_match_table1'])|set(config['ik_match_table2'])
 missing=[x for x in bodies if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,x)<0]
 if missing:raise ValueError('GMR target frames missing: '+str(missing))
 differing=[]
 for name in names:
  j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name); low,high=map(float,model.jnt_range[j]);u=limits[name]
  if abs(low-u[0])>1e-6 or abs(high-u[1])>1e-6:differing.append({'name':name,'gmr':[low,high],'dreamcontrol_urdf':list(u)})
 if any(sha(Path(p))!=h for p,h in pins.items()):raise RuntimeError('source changed')
 report={'model_sha256':sha(target),'source_sha256':pins[str(SOURCE)],'urdf_sha256':pins[str(URDF)],'gmr_config_sha256':pins[str(CONFIG)],'joint_names':names,'removed_joints':removed,'nq':model.nq,'nv':model.nv,'nu':model.nu,'declared_limit_differences':differing,'training_ready':False,'limitations':['Modified GMR model using upstream 29-DoF geometry with two joints fixed at zero.','Body-frame FK compatibility, scaling, foot contact, dynamics and actual GMR output require independent checks.']}
 with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
 print(json.dumps({'nq':model.nq,'nv':model.nv,'nu':model.nu,'declared_limit_differences':len(differing),'training_ready':False}))
if __name__=='__main__':main()

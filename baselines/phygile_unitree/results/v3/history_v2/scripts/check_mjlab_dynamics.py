"""Check compiled official G1 actuator parameters; no policy/simulation steps."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--unitree-checkout',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.unitree_checkout));from mjlab_task import make_config
import mujoco
c=json.loads((a.assets/'contract.json').read_text(encoding='utf-8'))
cfg=make_config(c,'unused-not-loaded.npz',149);entity=cfg.scene.entities['robot'].build();model=entity.compile()
rows=[]
for i in range(model.nu):
 j=int(model.actuator_trnid[i,0]);name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j);k=c['joint_names'].index(name);d=int(model.jnt_dofadr[j])
 observed={'stiffness':float(model.actuator_gainprm[i,0]),'damping':float(-model.actuator_biasprm[i,2]),'joint_armature':float(model.dof_armature[d]),'joint_effort_limits':float(model.actuator_forcerange[i,1])}
 for field,value in observed.items():
  if value!=c[field][k]:raise RuntimeError('Compiled dynamics contract mismatch '+name+' '+field)
 rows.append({'joint':name,**observed,'passive_damping':float(model.dof_damping[d]),'frictionloss':float(model.dof_frictionloss[d])})
if len(rows)!=29:raise RuntimeError('Wrong actuator count')
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
result={'status':'COMPILED_PD_EFFORT_ARMATURE_EXACT','actual_physics_steps':0,'actuators':rows,'official_g1_xml_sha256':hashlib.sha256(G1_XML.read_bytes()).hexdigest(),'scope':'Static compiled official G1 parameter check; complements actual stepping evaluations; does not assert PhysX/MuJoCo solver or geometry equivalence.'}
with a.output.open('x',encoding='utf-8') as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

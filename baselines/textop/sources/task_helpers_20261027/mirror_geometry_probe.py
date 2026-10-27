import pathlib,subprocess,os
py='/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python'
code=r'''
import mujoco,numpy as np,json
from scipy.spatial.transform import Rotation as R
m=mujoco.MjModel.from_xml_path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml');S=np.diag([1,-1,1]);swap=lambda s:s.replace('left_','TMP_').replace('right_','left_').replace('TMP_','right_')
rows=[];bounds={0:0};rotmax=0
for i in range(1,m.nbody):
    name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i);j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,swap(name))
    pos=np.linalg.norm(m.body_pos[i]-S@m.body_pos[j]);qi=m.body_quat[i];qj=m.body_quat[j]
    rr=np.abs(R.from_quat(qi[[1,2,3,0]]).as_matrix()-S@R.from_quat(qj[[1,2,3,0]]).as_matrix()@S).max();rotmax=max(rotmax,float(rr));bounds[i]=bounds[int(m.body_parentid[i])]+float(pos)
    if pos>1e-8 or rr>1e-8:rows.append({'body':name,'local_pos_asym_m':float(pos),'local_rot_asym':float(rr),'chain_bound_m':bounds[i]})
print(json.dumps({'local_asymmetries':rows,'maximum_chain_bound_m':max(bounds.values()),'maximum_local_rot_asym':rotmax}))
'''
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
p=subprocess.run([py,'-c',code],capture_output=True,text=True,env=env);print(p.stdout,p.stderr)

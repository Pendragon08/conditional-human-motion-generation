"""Extract sampled-state force-balance inputs from an explicit MuJoCo model.

Experimental plumbing only: callers must supply verified qpos/qvel/qacc,
joint effort bounds, contact positions, orthonormal bases, friction and model
provenance. No contacts are detected or inferred, and no dataset is accepted.
"""
import numpy as np
import mujoco
from contact_dynamics_feasibility import check_force_balance


def evaluate(model, qpos, qvel, qacc, *, actuated_joint_names, torque_lower,
             torque_upper, contacts, tolerance):
    qpos=np.asarray(qpos,dtype=float);qvel=np.asarray(qvel,dtype=float);qacc=np.asarray(qacc,dtype=float)
    if (qpos.shape!=(model.nq,) or qvel.shape!=(model.nv,) or qacc.shape!=(model.nv,) or
        not np.isfinite(qpos).all() or not np.isfinite(qvel).all() or not np.isfinite(qacc).all()):
        raise ValueError('invalid MuJoCo generalized state')
    normalized=qpos.copy();mujoco.mj_normalizeQuat(model,normalized)
    if not np.allclose(qpos,normalized,atol=1e-7,rtol=0):raise ValueError('unnormalized quaternion')
    if (not isinstance(actuated_joint_names,list) or len(set(actuated_joint_names))!=len(actuated_joint_names) or
        not isinstance(contacts,list)):
        raise ValueError('explicit unique actuator joints and contact list required')
    data=mujoco.MjData(model);data.qpos[:]=qpos;data.qvel[:]=qvel
    mujoco.mj_forward(model,data)
    mass=np.empty((model.nv,model.nv),dtype=np.float64);mujoco.mj_fullM(model,mass,data.qM)
    # Fresh MjData has no applied external loads or actuator control. Passive
    # joint forces (e.g. damping and springs) act on the right-hand side.
    bias=np.array(data.qfrc_bias-data.qfrc_passive,copy=True)
    act=np.zeros((model.nv,len(actuated_joint_names)))
    for col,name in enumerate(actuated_joint_names):
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if jid<0 or model.jnt_type[jid] not in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError('actuation requires named scalar hinge or slide joint')
        act[int(model.jnt_dofadr[jid]),col]=1.
    jac=[];friction=[]
    for contact in contacts:
        if not isinstance(contact,dict):raise ValueError('explicit contact dictionaries required')
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,contact.get('body',''))
        point=np.asarray(contact.get('point_world'),dtype=float)
        basis=np.asarray(contact.get('basis_world'),dtype=float)
        if (body<=0 or point.shape!=(3,) or basis.shape!=(3,3) or
            not np.isfinite(point).all() or not np.isfinite(basis).all() or
            not np.allclose(basis.T@basis,np.eye(3),rtol=0,atol=1e-8) or
            abs(np.linalg.det(basis)-1)>1e-8):
            raise ValueError('invalid contact body, point or right-handed basis')
        j=np.zeros((3,model.nv));mujoco.mj_jac(model,data,j,None,point,body)
        jac.append(basis.T@j);friction.append(contact.get('friction'))
    jac=np.asarray(jac,dtype=float).reshape((-1,3,model.nv))
    result=check_force_balance(mass,qacc,bias,act,jac,torque_lower=torque_lower,
                               torque_upper=torque_upper,friction=friction,tolerance=tolerance)
    result['model_nq']=int(model.nq);result['model_nv']=int(model.nv)
    result['contact_count']=len(contacts);result['actuated_joints']=list(actuated_joint_names)
    result['limitations']=(result.get('limitations',[])+[
        'Contacts, state, model, torque limits and friction are caller declarations, not validated measurements.',
        'Passive forces are included; unspecified external loads are assumed zero.',
        'MuJoCo extraction is not an Isaac Lab runtime or full trajectory validation.'])
    return result

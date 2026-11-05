"""Per-foot kinematic ground-contact flags from MuJoCo's collision query.

Flags represent generated contacts with the scene floor, not measured force or
true source labels. Poses are set directly and velocities are zero; no dynamics.
"""
import numpy as np
import mujoco


def _id(model, kind, name):
    result=mujoco.mj_name2id(model,kind,name)
    if result<0: raise ValueError(f'missing MuJoCo object {name}')
    return result


def flags(model, motion):
    pos=np.asarray(motion['root_pos']); quat=np.asarray(motion['root_quat_xyzw']); joints=np.asarray(motion['joint_pos'])
    names=tuple(motion['joint_names']); frames=len(pos)
    if (pos.shape!=(frames,3) or quat.shape!=(frames,4) or joints.shape!=(frames,len(names)) or
            frames<2 or len(set(names))!=len(names) or not np.isfinite(pos).all() or
            not np.isfinite(quat).all() or not np.isfinite(joints).all()):
        raise ValueError('invalid motion arrays or joint names')
    root=_id(model,mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint')
    if int(model.jnt_type[root])!=int(mujoco.mjtJoint.mjJNT_FREE): raise ValueError('expected floating root')
    qaddr=int(model.jnt_qposadr[root]); floor=_id(model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    feet=[_id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in ('left_ankle_roll_link','right_ankle_roll_link')]
    joint_ids=[_id(model,mujoco.mjtObj.mjOBJ_JOINT,name) for name in names]
    data=mujoco.MjData(model); result=np.zeros((frames,2),dtype=bool)
    for frame in range(frames):
        mujoco.mj_resetData(model,data)
        data.qpos[qaddr:qaddr+3]=pos[frame]
        data.qpos[qaddr+3:qaddr+7]=quat[frame,[3,0,1,2]]
        for j,value in zip(joint_ids,joints[frame]): data.qpos[model.jnt_qposadr[j]]=value
        data.qvel[:]=0; mujoco.mj_forward(model,data)
        for k in range(data.ncon):
            a,b=int(data.contact[k].geom1),int(data.contact[k].geom2)
            if floor in (a,b):
                body=int(model.geom_bodyid[b if a==floor else a])
                for side,foot in enumerate(feet):
                    if body==foot: result[frame,side]=True
    return result


def confusion(source, target):
    source=np.asarray(source); target=np.asarray(target)
    if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=2 or source.dtype!=bool or target.dtype!=bool:
        raise ValueError('expected matching boolean (T,2) flags')
    tp=int(np.sum(source & target)); fn=int(np.sum(source & ~target))
    fp=int(np.sum(~source & target)); tn=int(np.sum(~source & ~target))
    return {'tp':tp,'fn':fn,'fp':fp,'tn':tn,'recall':tp/(tp+fn) if tp+fn else None,
            'precision':tp/(tp+fp) if tp+fp else None,'agreement':(tp+tn)/(tp+fn+fp+tn)}

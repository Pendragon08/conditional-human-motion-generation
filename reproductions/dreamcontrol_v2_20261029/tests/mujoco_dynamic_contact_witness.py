"""Short CPU MuJoCo contact/dynamics diagnostic for one G1 motion.

Uses existing DreamControl simulation scene and gains, not a learned policy or
Isaac Lab. No task object is instantiated. Initial base/joint velocities are
zero; a simple joint PD controller follows reference joint positions while the
floating root is unconstrained. This is not an acceptance or success test.
"""
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import zipfile
import numpy as np
import torch
import yaml
import mujoco
from g1_fk import URDFFK
from omniretarget_bridge import convert_layout
from resample_omniretarget import resample
from waist_lock_ik import FIT_LINKS
from waist_lock_root_ik import fit_root_and_waist


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def name_id(model,kind,name):
    index=mujoco.mj_name2id(model,kind,name)
    if index<0: raise ValueError(f'missing MuJoCo {kind}: {name}')
    return index


def simulate(scene, gains, urdf, fit, seconds=1.0):
    model=mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep=.002
    data=mujoco.MjData(model)
    cfg=yaml.safe_load(gains.read_text())
    names=[n[:-5]+'_joint' if n.startswith(('left_hand_','right_hand_')) and n.endswith('_link') else n
           for n in cfg['joint_names']]
    if len(names)!=len(cfg['kps']) or len(names)!=len(cfg['kds']) or len(set(names))!=len(names):
        raise ValueError('invalid controller config')
    ids=[name_id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
    aids=[name_id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in names]
    if len(names)!=model.nu or len(set(aids))!=model.nu:
        raise ValueError('controller must cover all actuators exactly')
    body_names=list(fit['joint_names'])
    if len(body_names)!=27 or not set(body_names)<=set(names): raise ValueError('expected 27 body joints')
    order={name:i for i,name in enumerate(body_names)}
    initial=fit['root_quat_xyzw'][0]
    root_joint=name_id(model,mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint')
    root_qaddr=int(model.jnt_qposadr[root_joint]); root_vaddr=int(model.jnt_dofadr[root_joint])
    if int(model.jnt_type[root_joint])!=int(mujoco.mjtJoint.mjJNT_FREE): raise ValueError('expected free root')
    data.qpos[root_qaddr:root_qaddr+3]=fit['root_pos'][0]
    data.qpos[root_qaddr+3:root_qaddr+7]=initial[[3,0,1,2]]
    for n,j in zip(names,ids):
        data.qpos[model.jnt_qposadr[j]]=fit['joint_pos'][0,order[n]] if n in order else 0.
    data.qvel[:]=0
    mujoco.mj_forward(model,data)
    body_ids=[name_id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in FIT_LINKS]
    expected=URDFFK(urdf).forward(fit['root_pos'][:1],fit['root_quat_xyzw'][:1],fit['joint_pos'][:1],body_names,links=FIT_LINKS)[0]
    actual=data.xpos[body_ids].copy()
    fk_max=float(np.linalg.norm(expected-actual,axis=1).max())
    if fk_max>1e-3: raise ValueError(f'URDF vs MuJoCo 12-link position mismatch: {fk_max} m')
    floor=name_id(model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    feet={name_id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in ('left_ankle_roll_link','right_ankle_roll_link')}
    kp=np.asarray(cfg['kps'],dtype=float); kd=np.asarray(cfg['kds'],dtype=float)
    if not np.isfinite(kp).all() or not np.isfinite(kd).all() or np.any(kp<0) or np.any(kd<0):
        raise ValueError('invalid joint gains')
    total_steps=min(int(seconds/model.opt.timestep),int((len(fit['joint_pos'])-1)/20/model.opt.timestep))
    if total_steps<1: raise ValueError('reference too short')
    foot_contact_steps=0; nonfoot_contact_steps=0; self_contact_steps=0; max_force=0.; max_pen=0.
    root_z_min=float('inf'); root_error=[]; joint_error=[]; saturation=0; force_count=0
    nonfoot_bodies=Counter(); self_pairs=Counter(); first_nonfoot_time=None
    for step in range(total_steps):
        t=step*model.opt.timestep; frame=min(int(t*20),len(fit['joint_pos'])-2); alpha=t*20-frame
        ref=(1-alpha)*fit['joint_pos'][frame]+alpha*fit['joint_pos'][frame+1]
        root_ref=(1-alpha)*fit['root_pos'][frame]+alpha*fit['root_pos'][frame+1]
        errors=[]
        for k,(n,j,a) in enumerate(zip(names,ids,aids)):
            target=float(ref[order[n]]) if n in order else 0.
            position=float(data.qpos[model.jnt_qposadr[j]]); velocity=float(data.qvel[model.jnt_dofadr[j]])
            control=kp[k]*(target-position)-kd[k]*velocity
            if model.jnt_actfrclimited[j]:
                lo,hi=model.jnt_actfrcrange[j]; bounded=float(np.clip(control,lo,hi)); saturation+=int(bounded!=control); control=bounded
            data.ctrl[a]=control
            if n in order: errors.append(position-target)
        mujoco.mj_step(model,data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all(): raise ValueError('non-finite simulated state')
        if not np.isclose(data.time,(step+1)*model.opt.timestep,atol=1e-9,rtol=0):
            raise ValueError('simulator time changed unexpectedly, possible numerical reset')
        z=float(data.qpos[root_qaddr+2]); root_z_min=min(root_z_min,z)
        root_error.append(float(np.linalg.norm(data.qpos[root_qaddr:root_qaddr+3]-root_ref)))
        joint_error.append(float(np.sqrt(np.mean(np.square(errors)))))
        foot_here=False; nonfoot_here=False; self_here=False
        for ci in range(data.ncon):
            c=data.contact[ci]; g1,g2=int(c.geom1),int(c.geom2)
            if floor in (g1,g2):
                other=g2 if g1==floor else g1; foot_here |= int(model.geom_bodyid[other]) in feet
                body_id=int(model.geom_bodyid[other]); nonfoot_here |= body_id not in feet
                if body_id not in feet:
                    nonfoot_bodies[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body_id) or str(body_id)]+=1
                max_pen=max(max_pen,max(0.,-float(c.dist)))
                force=np.zeros(6); mujoco.mj_contactForce(model,data,ci,force)
                max_force=max(max_force,float(force[0])); force_count+=1
            else:
                self_here=True
                bodies=sorted(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or str(int(model.geom_bodyid[g])) for g in (g1,g2))
                self_pairs[' | '.join(bodies)]+=1
        foot_contact_steps+=int(foot_here); nonfoot_contact_steps+=int(nonfoot_here); self_contact_steps+=int(self_here)
        if nonfoot_here and first_nonfoot_time is None: first_nonfoot_time=float(data.time)
    return {'fps':20,'sim_dt':model.opt.timestep,'gravity':model.opt.gravity.tolist(),'steps':total_steps,
            'simulated_seconds':total_steps*model.opt.timestep,'fk_initial_max_error_m':fk_max,
            'foot_floor_contact_step_fraction':foot_contact_steps/total_steps,
            'nonfoot_floor_contact_step_fraction':nonfoot_contact_steps/total_steps,
            'self_contact_step_fraction':self_contact_steps/total_steps,'max_floor_contact_depth_m':max_pen,
            'max_normal_force_N':max_force,'floor_contact_measurements':force_count,'root_min_height_m':root_z_min,
            'first_nonfoot_ground_time_s':first_nonfoot_time,'nonfoot_ground_contact_counts':dict(nonfoot_bodies),
            'self_contact_pair_counts':dict(self_pairs),
            'mean_root_position_error_m':float(np.mean(root_error)),'mean_joint_rms_error_rad':float(np.mean(joint_error)),
            'control_saturation_fraction':saturation/(total_steps*len(names)),
            'initial_base_and_joint_velocities':'zero','controller':'joint-position PD; no learned balance or object interaction'}


def main():
    torch.set_num_threads(1)
    root=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
    project=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
    assets=project/'Training/HumanoidVerse/humanoidverse/data/robots/g1'
    sim_assets=project/'Sim2Real/resources/robots/g1_description'
    scene=sim_assets/'scene_full.xml'; model_xml=sim_assets/'g1_27dof_with_hand_rev_1_0.xml'
    gains=project/'Sim2Real/deploy/deploy_mujoco/configs/g1_full_body.yaml'
    u29,u27=assets/'g1_29dof.urdf',assets/'g1_27dof.urdf'
    member='robot-object/sub8_largebox_024_original.npz'; archive_path=root/'data/OmniRetarget_Dataset/robot-object.zip'
    with zipfile.ZipFile(archive_path) as archive:
        with np.load(io.BytesIO(archive.read(member)),allow_pickle=False) as data:
            q=resample(data['qpos'],float(data['fps']))[:200]
    motion,_=convert_layout(q); results=[]
    for weight in (0.,10.):
        fit=fit_root_and_waist(motion,u29,u27,steps=120,learning_rate=.015,foot_position_weight=2.,
                               contact_velocity_weight=10.,ground_penetration_weight=weight,ground_plane_z=0.)
        results.append({'ground_weight':weight,'metrics':simulate(scene,gains,u27,fit)})
    print(json.dumps({'archive':archive_path.name,'member':member,'results':results,'mujoco_version':mujoco.__version__,
                      'input_hashes':{p.name:digest(p) for p in (scene,model_xml,gains,u27,u29,archive_path)},
                      'warning':'One clip, first second only; existing MuJoCo scene flat ground with gravity -7.5 m/s2 and 14 hand joints, no task object, no trained balance. Source floor not verified; not Isaac Lab, physical dataset acceptance or paper replication.'},allow_nan=False))


if __name__=='__main__': main()

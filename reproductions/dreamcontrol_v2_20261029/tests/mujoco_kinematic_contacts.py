"""Full-clip MuJoCo collision query of six previously selected box motions.

Frames are kinematically placed, velocities zeroed, and mj_forward computes
geometric contact. No time integration, real measured terrain or box is used;
therefore this is not a dynamic trajectory validation or a training pass.
"""
from collections import Counter
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
import mujoco
from mujoco_dynamic_contact_witness import digest,name_id
from omniretarget_bridge import convert_layout
from resample_omniretarget import resample
from waist_lock_root_ik import fit_root_and_waist


def scan(model, fit):
    data=mujoco.MjData(model)
    root=name_id(model,mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint')
    if int(model.jnt_type[root])!=int(mujoco.mjtJoint.mjJNT_FREE): raise ValueError('expected floating root')
    root_q=int(model.jnt_qposadr[root]); floor=name_id(model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    feet={name_id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in ('left_ankle_roll_link','right_ankle_roll_link')}
    body_names=fit['joint_names']
    joint_ids=[name_id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in body_names]
    feet_frames=nonfoot_frames=self_frames=0; max_depth=0.; min_floor_dist=None
    nonfoot_counts=Counter(); self_counts=Counter()
    for t in range(len(fit['joint_pos'])):
        mujoco.mj_resetData(model,data)
        data.qpos[root_q:root_q+3]=fit['root_pos'][t]
        data.qpos[root_q+3:root_q+7]=fit['root_quat_xyzw'][t][[3,0,1,2]]
        for k,j in enumerate(joint_ids): data.qpos[model.jnt_qposadr[j]]=fit['joint_pos'][t,k]
        data.qvel[:]=0
        mujoco.mj_forward(model,data)
        if not np.isfinite(data.qpos).all(): raise ValueError('non-finite static state')
        ft=nft=sf=False
        for ci in range(data.ncon):
            c=data.contact[ci]; a,b=int(c.geom1),int(c.geom2)
            if floor in (a,b):
                other=b if a==floor else a; body=int(model.geom_bodyid[other])
                ft |= body in feet; nft |= body not in feet
                distance=float(c.dist); max_depth=max(max_depth,max(0.,-distance))
                min_floor_dist=distance if min_floor_dist is None else min(min_floor_dist,distance)
                if body not in feet:
                    nonfoot_counts[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body) or str(body)]+=1
            else:
                sf=True
                labels=sorted(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or str(int(model.geom_bodyid[g])) for g in (a,b))
                self_counts[' | '.join(labels)]+=1
        feet_frames+=int(ft); nonfoot_frames+=int(nft); self_frames+=int(sf)
    n=len(fit['joint_pos'])
    return {'frames':n,'foot_floor_contact_frame_fraction':feet_frames/n,'nonfoot_floor_contact_frame_fraction':nonfoot_frames/n,
            'self_contact_frame_fraction':self_frames/n,'max_floor_contact_depth_m':max_depth,
            'min_floor_contact_distance_m':min_floor_dist,'nonfoot_floor_contact_counts':dict(nonfoot_counts),
            'self_contact_pair_counts':dict(self_counts)}


def main():
    torch.set_num_threads(1)
    root=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
    project=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
    assets=project/'Training/HumanoidVerse/humanoidverse/data/robots/g1'
    sim_assets=project/'Sim2Real/resources/robots/g1_description'
    scene=sim_assets/'scene_full.xml'; model_xml=sim_assets/'g1_27dof_with_hand_rev_1_0.xml'
    u29,u27=assets/'g1_29dof.urdf',assets/'g1_27dof.urdf'
    archive_path=root/'data/OmniRetarget_Dataset/robot-object.zip'
    selection=json.loads((root/'foot_ground_fit_20261030/holdout_report.json').read_text())['samples']
    if len(selection)!=6: raise ValueError('expected exactly six preselected clips')
    model=mujoco.MjModel.from_xml_path(str(scene)); outputs=[]
    with zipfile.ZipFile(archive_path) as archive:
        for item in selection:
            with np.load(io.BytesIO(archive.read(item['member'])),allow_pickle=False) as data:
                q=resample(data['qpos'],float(data['fps']))[:200]
            motion,_=convert_layout(q); methods=[]
            for k,weight in enumerate((0.,10.)):
                fit=fit_root_and_waist(motion,u29,u27,steps=120,learning_rate=.015,foot_position_weight=2.,
                                       contact_velocity_weight=10.,ground_penetration_weight=weight,ground_plane_z=0.)
                if abs(fit['final_max_keypoint_error_m']-item['configs'][k]['max_keypoint_error_m'])>1e-8:
                    raise ValueError('recomputed fit differs from prior selection report')
                methods.append({'ground_weight':weight,'max_keypoint_error_m':fit['final_max_keypoint_error_m'],'metrics':scan(model,fit)})
            outputs.append({'subject':item['subject'],'member':item['member'],'methods':methods})
    print(json.dumps({'samples':outputs,'mujoco_version':mujoco.__version__,
                      'input_hashes':{p.name:digest(p) for p in (scene,model_xml,u27,u29,archive_path)},
                      'warning':'Kinematic poses only, zero velocities, source flat z=0 unverified, no object. Contacts reflect MuJoCo filters, not Isaac Lab, forces or dynamics; these are not accepted training data.'},allow_nan=False))


if __name__=='__main__': main()

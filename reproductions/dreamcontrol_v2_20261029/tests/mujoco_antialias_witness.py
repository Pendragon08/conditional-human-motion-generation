"""Compare linear and FIR resampling in a limited CPU contact simulation.

Both methods run identical experimental fitting and simple joint PD. The
simulation's flat ground, -7.5 gravity, zero initial velocity and lack of task
object/balance policy limit interpretation; this is not a data-quality pass.
"""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
import mujoco
from mujoco_dynamic_contact_witness import simulate,digest
from omniretarget_bridge import convert_layout
from resample_antialias_omniretarget import resample_antialias
from resample_omniretarget import resample
from waist_lock_root_ik import fit_root_and_waist


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
            original=data['qpos'].copy(); fps=float(data['fps'])
    sampled=[resample(original,fps)[:200],resample_antialias(original,fps)[:200]]
    if sampled[0].shape!=sampled[1].shape: raise ValueError('different output sample shapes')
    delta=np.linalg.norm(sampled[0][:,4:7]-sampled[1][:,4:7],axis=-1)
    results=[]
    for label,q in zip(('linear','fir'),sampled):
        motion,_=convert_layout(q)
        fit=fit_root_and_waist(motion,u29,u27,steps=120,learning_rate=.015,foot_position_weight=2.,
                               contact_velocity_weight=10.,ground_penetration_weight=10.,ground_plane_z=0.)
        results.append({'resampling':label,'fit_mean_error_m':fit['final_mean_keypoint_error_m'],
                        'fit_max_error_m':fit['final_max_keypoint_error_m'],'metrics':simulate(scene,gains,u27,fit)})
    print(json.dumps({'archive':archive_path.name,'member':member,'results':results,
                      'mean_root_position_difference_m':float(delta.mean()),'max_root_position_difference_m':float(delta.max()),
                      'mujoco_version':mujoco.__version__,'input_hashes':{p.name:digest(p) for p in (scene,model_xml,gains,u27,u29,archive_path)},
                      'warning':'One clip and one second only; flat z=0 ground not proven for source; -7.5 m/s2, zero initial velocity, 14 hand joints, no object or trained balance. No Isaac Lab, physical quality acceptance or true-error comparison.'},allow_nan=False))


if __name__=='__main__': main()

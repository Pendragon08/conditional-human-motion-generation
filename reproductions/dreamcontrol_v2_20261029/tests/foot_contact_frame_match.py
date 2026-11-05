"""Per-foot/source-frame contact confusion for fixed baseline and candidate.

Static MuJoCo floor collision flags are compared to source-projected foot flags.
No task object, force, source ground confirmation or dynamics are evaluated.
"""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
import mujoco
from motion_bridge import SOURCE_NAMES
from mujoco_contact_flags import flags,confusion
from mujoco_dynamic_contact_witness import digest
from omniretarget_bridge import convert_layout
from resample_omniretarget import resample
from waist_lock_root_ik import fit_root_and_waist


def main():
    torch.set_num_threads(1)
    root=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
    selected_file=root/'foot_contact_fit_20261030/holdout_report.json'
    source_file=root/'antialias_mujoco_20261030/source_report.json'
    selected_report=json.loads(selected_file.read_text())
    selected=selected_report['samples']
    source_rows={x['member']:x for x in json.loads(source_file.read_text())['samples']}
    if len(selected)!=6 or len(source_rows)!=6: raise ValueError('expected six unique clips')
    project=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
    assets=project/'Training/HumanoidVerse/humanoidverse/data/robots/g1'
    scene=project/'Sim2Real/resources/robots/g1_description/scene_full.xml'
    model_xml=scene.parent/'g1_27dof_with_hand_rev_1_0.xml'
    u29,u27=assets/'g1_29dof.urdf',assets/'g1_27dof.urdf'
    archive_path=root/'data/OmniRetarget_Dataset/robot-object.zip'
    recorded=selected_report['input_hashes']
    for name,path in (('scene_sha256',scene),('urdf27_sha256',u27),('urdf29_sha256',u29),('archive_sha256',archive_path)):
        if digest(path)!=recorded[name]: raise ValueError(f'input changed since prior report: {name}')
    model=mujoco.MjModel.from_xml_path(str(scene)); rows=[]
    names=[n for n in SOURCE_NAMES if n not in ('waist_roll_joint','waist_pitch_joint')]
    indices=[SOURCE_NAMES.index(n) for n in names]
    with zipfile.ZipFile(archive_path) as archive:
        for item in selected:
            member=item['member']
            if member not in source_rows: raise ValueError('missing source baseline')
            with np.load(io.BytesIO(archive.read(member)),allow_pickle=False) as data:
                q=resample(data['qpos'],float(data['fps']))[:200]
            motion,_=convert_layout(q)
            nominal={'root_pos':motion[:,:3],'root_quat_xyzw':motion[:,3:7],
                     'joint_pos':motion[:,7:][:,indices],'joint_names':names}
            source=flags(model,nominal)
            if abs(np.any(source,axis=1).mean()-source_rows[member]['metrics']['foot_floor_contact_frame_fraction'])>1e-12:
                raise ValueError('source contact count differs from earlier report')
            methods=[]
            for height in (0.,10.):
                fit=fit_root_and_waist(motion,u29,u27,steps=120,learning_rate=.015,foot_position_weight=2.,
                                       contact_velocity_weight=10.,ground_penetration_weight=10.,ground_plane_z=0.,
                                       ground_contact_weight=height,ground_contact_band_m=.005)
                earlier=item['baseline'] if height==0 else item['candidate']
                if abs(fit['final_max_keypoint_error_m']-earlier['max_keypoint_error_m'])>1e-8:
                    raise ValueError('fitted motion differs from previous report')
                found=flags(model,fit)
                earlier_stats=earlier['metrics'] if height==0 else earlier['sim_contacts']
                if abs(np.any(found,axis=1).mean()-earlier_stats['foot_floor_contact_frame_fraction'])>1e-12:
                    raise ValueError('contact count differs from previous report')
                methods.append({'height_weight':height,'per_foot_frame_confusion':confusion(source,found)})
            rows.append({'subject':item['subject'],'member':member,'methods':methods})
    print(json.dumps({'samples':rows,'mujoco_version':mujoco.__version__,
                      'input_hashes':{p.name:digest(p) for p in (selected_file,source_file,scene,model_xml,u29,u27,archive_path)},
                      'warning':'Same six box paths. Source ground z=0 unconfirmed, projected upper body differs, static MuJoCo collision flags are not measured contact forces; no object or dynamics. Do not claim dataset acceptance.'},allow_nan=False))


if __name__=='__main__': main()

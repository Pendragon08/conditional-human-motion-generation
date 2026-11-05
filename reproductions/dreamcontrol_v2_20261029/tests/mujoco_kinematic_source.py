"""Compare original lower-body feet to a nominal 27-DoF projection.

Static MuJoCo collision queries only: no dynamics, no task object, unknown
actual source ground. Source feet correspondence is checked by URDF FK.
"""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
import mujoco
from g1_fk import URDFFK
from motion_bridge import SOURCE_NAMES
from mujoco_dynamic_contact_witness import digest
from mujoco_kinematic_contacts import scan
from omniretarget_bridge import convert_layout
from resample_omniretarget import resample


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
    if len(selection)!=6: raise ValueError('expected six selected clips')
    fk29,fk27=URDFFK(u29),URDFFK(u27)
    links=('left_ankle_roll_link','right_ankle_roll_link')
    names=[name for name in SOURCE_NAMES if name not in ('waist_pitch_joint','waist_roll_joint')]
    indices=[SOURCE_NAMES.index(name) for name in names]
    model=mujoco.MjModel.from_xml_path(str(scene)); outputs=[]
    with zipfile.ZipFile(archive_path) as archive:
        for item in selection:
            with np.load(io.BytesIO(archive.read(item['member'])),allow_pickle=False) as data:
                q=resample(data['qpos'],float(data['fps']))[:200]
            motion,_=convert_layout(q)
            pos,quat=motion[:,:3],motion[:,3:7]
            p29,r29=fk29.forward(pos,quat,motion[:,7:],SOURCE_NAMES,links=links,return_rotations=True)
            joints=motion[:,7:][:,indices]
            p27,r27=fk27.forward(pos,quat,joints,names,links=links,return_rotations=True)
            position_error=float(np.linalg.norm(p29-p27,axis=-1).max())
            rotation_error=float(np.max(np.abs(r29-r27)))
            if position_error>1e-7 or rotation_error>1e-7: raise ValueError('source and projected foot FK differ')
            nominal={'root_pos':pos,'root_quat_xyzw':quat,'joint_names':names,'joint_pos':joints}
            outputs.append({'subject':item['subject'],'member':item['member'],
                            'max_foot_position_diff_m':position_error,'max_foot_rotation_matrix_element_diff':rotation_error,
                            'metrics':scan(model,nominal)})
    print(json.dumps({'samples':outputs,'mujoco_version':mujoco.__version__,
                      'input_hashes':{p.name:digest(p) for p in (scene,model_xml,u27,u29,archive_path)},
                      'warning':'Source feet are FK-matched to a nominal 27-DoF projection; upper-body pose is not source-equivalent. Static queries, zero velocities, no box, assumed ground and MuJoCo filters. No dynamics or physical acceptance.'},allow_nan=False))


if __name__=='__main__': main()

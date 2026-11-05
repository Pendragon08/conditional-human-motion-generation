"""Drake static self-collision diagnostic on a fitted 27-DoF box clip.

Uses URDF geometries and Drake's default collision filtering, not Isaac Lab's
configuration. There is no ground, manipulated object, simulator contact or
dynamics in this diagnostic. The limited frames do not validate a trajectory.
"""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.parsing import Parser
from motion_bridge import SOURCE_NAMES
from omniretarget_bridge import convert_layout
from resample_omniretarget import resample
from waist_lock_root_ik import fit_root_and_waist
from prepare_drake_collision_model import sha256


def main():
    torch.set_num_threads(1)
    root=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
    assets=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')
    u29,u27=assets/'g1_29dof.urdf',assets/'g1_27dof.urdf'
    member='robot-object/sub3_largebox_015_original.npz'
    with zipfile.ZipFile(root/'data/OmniRetarget_Dataset/robot-object.zip') as archive:
        with np.load(io.BytesIO(archive.read(member)),allow_pickle=False) as data:
            q=resample(data['qpos'],float(data['fps']))[:200]
    motion,_=convert_layout(q)
    fit=fit_root_and_waist(motion,u29,u27,steps=120,learning_rate=.015,foot_position_weight=2.,contact_velocity_weight=10.)
    builder=DiagramBuilder()
    plant,scene=AddMultibodyPlantSceneGraph(builder,time_step=0.)
    converted=root/'drake_collision_20261030/converted_geometry/robot_collision_obj.urdf'
    geometry=json.loads((converted.parent/'manifest.json').read_text())
    if (sha256(u27)!=geometry['source_urdf_sha256'] or sha256(converted)!=geometry['derived_urdf_sha256'] or
            any(sha256(item['source'])!=item['source_sha256'] or sha256(item['obj'])!=item['obj_sha256']
                for item in geometry['converted_meshes'])):
        raise ValueError('collision asset hash mismatch')
    models=Parser(plant).AddModels(str(converted.resolve()))
    if len(models)!=1: raise ValueError('expected one robot')
    plant.Finalize()
    names=list(plant.GetPositionNames(models[0])); expected=['pelvis_qw','pelvis_qx','pelvis_qy','pelvis_qz','pelvis_x','pelvis_y','pelvis_z']
    if names[:7]!=expected or set(names[7:])!={name+'_q' for name in fit['joint_names']}:
        raise ValueError('unexpected Drake position names')
    order={name:i for i,name in enumerate(fit['joint_names'])}
    diagram=builder.Build(); context=diagram.CreateDefaultContext()
    plant_context=plant.GetMyMutableContextFromRoot(context)
    scene_context=scene.GetMyContextFromRoot(context)
    rows=[]
    source_order={name:i for i,name in enumerate(SOURCE_NAMES)}
    for frame in sorted(set([0,len(motion)//2,len(motion)-1])):
        for method in ('nominal_27','fitted_27'):
            if method=='fitted_27':
                xyzw=fit['root_quat_xyzw'][frame]; pos=fit['root_pos'][frame]
                joints=[fit['joint_pos'][frame,order[name[:-2]]] for name in names[7:]]
            else:
                xyzw=motion[frame,3:7]; pos=motion[frame,:3]
                joints=[motion[frame,7+source_order[name[:-2]]] for name in names[7:]]
            q=np.concatenate((xyzw[[3,0,1,2]],pos,joints))
            plant.SetPositions(plant_context,models[0],q)
            query=scene.get_query_output_port().Eval(scene_context)
            inspector=query.inspector()
            pairs=query.ComputeSignedDistancePairwiseClosestPoints(0.0)
            penetrations=sorted([{'a':inspector.GetName(p.id_A),'b':inspector.GetName(p.id_B),
                                  'frame_a':inspector.GetName(inspector.GetFrameId(p.id_A)),
                                  'frame_b':inspector.GetName(inspector.GetFrameId(p.id_B)),
                                  'distance_m':float(p.distance)}
                                 for p in pairs if p.distance < 0],key=lambda x:x['distance_m'])
            rows.append({'frame':frame,'method':method,'penetrating_pair_count':len(penetrations),'pairs':penetrations})
    print(json.dumps({'archive':'robot-object.zip','member':member,'evaluated_frames':rows,
                      'converted_geometry_manifest':str(converted.parent/'manifest.json'),
                      'warning':'Static diagnostic using converted collision geometries and Drake default collision filters; no ground, object, contact response or dynamics; not an Isaac Lab or trajectory quality pass.'},allow_nan=False))


if __name__=='__main__': main()

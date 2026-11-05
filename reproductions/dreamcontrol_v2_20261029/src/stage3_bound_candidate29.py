"""Bound only sub-milliradian solver overshoot, then independently recompute every check.

This is an explicit generated-motion postprocessing operation, not a source-data
edit or acceptance-flag change. Large corrections are refused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
import pyroki as pk
import yourdfpy

S=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_pick29_20261103')
sys.path.insert(0,str(S.parent/'training_disk_guard_20261101'))
from g1_fk import URDFFK,PAPER_22_LINKS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--sample',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();assert not a.output_dir.exists()
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    original=json.loads((a.input_dir/'report.json').read_text())
    assert sha(a.input_dir/'robot29.npz')==original['output_sha256'] and sha(a.sample)==original['sample_sha256']
    with np.load(a.input_dir/'robot29.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    with np.load(a.sample,allow_pickle=False) as z:
        goal=z['target_points'];meta=json.loads(z['metadata_utf8'].tobytes())
    goal=goal@np.array(meta['canonical_to_world_rotation']).T+np.array(meta['canonical_to_world_translation'])
    urdf=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
    assert sha(urdf)==original['urdf_sha256']
    limit={j.get('name'):j.find('limit') for j in ET.parse(urdf).getroot().findall('joint') if j.get('type')=='revolute'}
    names=data['joint_names'].tolist()
    lo=np.array([float(limit[k].get('lower')) for k in names]);hi=np.array([float(limit[k].get('upper')) for k in names])
    bounded=np.clip(data['q'].astype(float),lo+1e-6,hi-1e-6).astype(np.float32)
    correction=float(np.max(np.abs(bounded-data['q'])))
    assert correction<.001,'large bound correction needs a new solve, not projection'
    data['q']=bounded
    fk=URDFFK(urdf);all_links=sorted(fk.links)
    points,rot=fk.forward(data['root_pos'],data['root_quat'][:,[1,2,3,0]],bounded,names,links=all_links,return_rotations=True)
    selected=[all_links.index(k) for k in PAPER_22_LINKS];feet=[all_links.index(k+'_ankle_roll_link') for k in ['left','right']]
    wrist=all_links.index('right_wrist_yaw_link')
    data['keypoints']=points[:,selected].astype(np.float32)
    data['right_wrist_quat']=Rotation.from_matrix(rot[:,wrist]).as_quat()[:,[3,0,1,2]].astype(np.float32)
    u=yourdfpy.URDF.load(str(urdf),load_meshes=False,load_collision_meshes=True)
    minima=[];foot_heights={}
    for k,name in enumerate(all_links):
        mesh=pk.collision.RobotCollision._get_trimesh_collision_geometries(u,name)
        v=np.asarray(mesh.vertices)
        if not len(v):continue
        z=(np.einsum('tij,vj->tvi',rot[:,k],v)+points[:,k,None])[:,:,2].min(axis=1)
        minima.append(z)
        if k in feet:foot_heights[k]=z
    minimum=float(np.stack(minima,axis=1).min());sole_max=float(np.stack(list(foot_heights.values()),axis=1).max())
    speed=np.linalg.norm(np.diff(points[:,feet,:2],axis=0),axis=-1)*20
    error=np.linalg.norm(points[:,wrist]-goal[:,21],axis=-1)
    initial=Rotation.from_matrix(rot[0,wrist]);desired=Rotation.from_quat([0.,2**-.5,0.,2**-.5])
    intended=Slerp([0,1.5],Rotation.from_quat(np.stack([initial.as_quat(),desired.as_quat()])))(np.minimum(data['sample_times'],1.5))
    angles=(Rotation.from_matrix(rot[:,wrist]).inv()*intended).magnitude()
    violation=float(max(0,np.max(lo-bounded),np.max(bounded-hi)))
    checks={'finite':bool(np.isfinite(points).all()),'joint_limits':violation<1e-7,
        'hand_target':bool(error.max()<.15),'wrist_orientation':bool(angles.max()<.35),
        'stance_foot_speed':bool(np.mean(speed>.15)<=.1),'ground_mesh_plane':minimum>=-.005,
        'feet_on_ground':sole_max<.02,'root_height':bool(np.all((data['root_pos'][:,2]>.3)&(data['root_pos'][:,2]<1.2)))}
    a.output_dir.mkdir(exist_ok=False);path=a.output_dir/'robot29.npz'
    with path.open('xb') as f:np.savez_compressed(f,**data)
    result={'status':'bounded_recomputed_candidate','input_state_sha256':original['output_sha256'],
        'input_report_sha256':sha(a.input_dir/'report.json'),'sample_sha256':sha(a.sample),
        'output_sha256':sha(path),'output':str(path),'urdf_sha256':sha(urdf),'scene_sha256':meta['scene_sha256'],
        'frames':len(bounded),'joint_count':29,'max_joint_projection_rad':correction,
        'max_joint_limit_violation_rad':violation,'max_hand_error_m':float(error.max()),
        'hand_error_at_grasp_m':float(error[40]),'max_wrist_angle_rad':float(angles.max()),
        'minimum_collision_mesh_height_m':minimum,'maximum_foot_sole_height_m':sole_max,
        'max_stance_foot_speed_mps':float(speed.max()),'checks':checks,
        'passed_requested_kinematic_checks':all(checks.values()),'native_scene_collision_replay':'pending',
        'eligible_for_calibrated_scene_reference':False,'no_source_dataset_filtering':True,
        'scope':'explicit tiny limit projection and independent NumPy/mesh/FK recheck; native replay still required'}
    with (a.output_dir/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

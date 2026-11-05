"""Align the isolated USD body's joint origins to the pinned training/postopt URDF.

The legacy locked-waist USD has three compensating upper-body offsets. Preserve
its mesh/inertia/hand assets, change only this project's copy, and verify in PhysX.
"""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'stage3_pick29_20261103'
sys.path.insert(0,str(R/'training_disk_guard_20261101'))
from g1_fk import URDFFK
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

source=D/'assets/g1_29dof_with_hands.usd'
target=D/'assets/g1_29dof_with_hands_urdf_aligned.usd'
assert not target.exists() and sha(source)=='cad66e43b85563903d535037a7e09051b9703092bdc216b2dd89102c010e1776'
urdf=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
assert sha(urdf)=='572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95'
stage=Usd.Stage.Open(str(source))
stage.Flatten().Export(str(target));stage=Usd.Stage.Open(str(target))
cache=UsdGeom.XformCache()
bodies={p.GetName():p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
root=bodies['pelvis'];root_matrix=cache.GetLocalToWorldTransform(root)
root_pos=np.asarray(root_matrix.ExtractTranslation(),dtype=float)
root_rot=np.asarray(root_matrix.ExtractRotationMatrix(),dtype=float).T
root_q=root_matrix.ExtractRotationQuat()
qxyzw=np.array([*root_q.GetImaginary(),root_q.GetReal()])[None]
fk=URDFFK(urdf)
common=sorted(set(bodies)&fk.links)
pos,rot=fk.forward(root_pos[None],qxyzw,np.zeros((1,29)),fk.actuated,links=common,return_rotations=True)


def matrix(position,rotation):
    m=np.eye(4);m[:3,:3]=rotation.T;m[3,:3]=position
    return Gf.Matrix4d(tuple(tuple(float(v) for v in row) for row in m))


new_world={name:matrix(pos[0,k],rot[0,k]) for k,name in enumerate(common)}
for name,p in bodies.items():
    if name in new_world:continue
    side='left' if name.startswith('left_hand_') else 'right' if name.startswith('right_hand_') else None
    if side is None:raise ValueError('unmapped physical body '+name)
    wrist=side+'_wrist_yaw_link'
    relative=cache.GetLocalToWorldTransform(p)*cache.GetLocalToWorldTransform(bodies[wrist]).GetInverse()
    new_world[name]=relative*new_world[wrist]
changed=[]
for name,p in bodies.items():
    before=cache.GetLocalToWorldTransform(p)
    parent=p.GetParent()
    # The flattened asset has all physical bodies directly under the articulation root.
    assert parent==root.GetParent(),str(p.GetPath())
    local=new_world[name]*cache.GetLocalToWorldTransform(parent).GetInverse()
    old_translation=np.asarray(before.ExtractTranslation(),dtype=float)
    error=float(np.max(np.abs(old_translation-np.asarray(new_world[name].ExtractTranslation()))))
    x=UsdGeom.Xformable(p);x.ClearXformOpOrder()
    x.AddTransformOp(opSuffix='training29_alignment').Set(local)
    if error>1e-6:changed.append({'body':name,'max_translation_change_m':error})
joints={j.get('name'):j for j in ET.parse(urdf).getroot().findall('joint')}
joint_changes=[]
for p in stage.Traverse():
    if not p.IsA(UsdPhysics.Joint) or p.GetName() not in joints:continue
    joint=UsdPhysics.Joint(p);origin=joints[p.GetName()].find('origin')
    xyz=np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
    old=np.asarray(joint.GetLocalPos0Attr().Get(),dtype=float)
    assert np.max(np.abs(np.asarray(joint.GetLocalPos1Attr().Get())))<1e-6
    if np.max(np.abs(old-xyz))>1e-6:
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(*map(float,xyz)))
        joint_changes.append({'joint':p.GetName(),'before':old.tolist(),'after':xyz.tolist()})
stage.GetRootLayer().Save()
report={'source_sha256':sha(source),'output_sha256':sha(target),'output':str(target),
        'urdf_sha256':sha(urdf),'body_changes':changed,'joint_origin_changes':joint_changes,
        'scope':'project asset adaptation; legacy collision meshes/inertias retained; runtime FK recheck required',
        'physical_runtime_validated':False}
with (D/'reports/asset29_urdf_alignment.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report))

"""Compare authored mass properties in a fixed waist cluster; no simulator."""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pxr import Usd, UsdPhysics
from scipy.spatial.transform import Rotation

BASE = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')
USD = BASE / 'g1_27dof_with_hands_min_collisions_flat_white.usd'
URDF = BASE / 'g1_27dof.urdf'
ANCHOR = 'waist_yaw_link'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def vec(s): return np.fromstring(s, sep=' ')
def quat(q):
    v=q.GetImaginary(); return Rotation.from_quat([*v,q.GetReal()]).as_matrix()
def matrix(R,p):
    t=np.eye(4); t[:3,:3]=R; t[:3,3]=p; return t
def aggregate(parts):
    m=sum(v['mass'] for v in parts); c=sum((v['mass']*v['center'] for v in parts),np.zeros(3))/m
    I=np.zeros((3,3))
    for v in parts:
        d=v['center']-c; I+=v['inertia']+v['mass']*((d@d)*np.eye(3)-np.outer(d,d))
    return {'mass_kg':m,'center_in_anchor_m':c.tolist(),'inertia_at_center_anchor_kg_m2':I.tolist()}

root=ET.parse(URDF).getroot(); links={p.get('name'):p for p in root.findall('link')}; stage=Usd.Stage.Open(str(USD))
edges=[]
for j in root.findall('joint'):
    if j.get('type')!='fixed': continue
    o=j.find('origin'); a=o.attrib if o is not None else {}
    edges.append((j.find('parent').get('link'), j.find('child').get('link'), matrix(Rotation.from_euler('xyz',vec(a.get('rpy','0 0 0'))).as_matrix(),vec(a.get('xyz','0 0 0')))))
urdf_pose={ANCHOR:np.eye(4)}
while True:
    before=len(urdf_pose)
    for a,b,t in edges:
        if a in urdf_pose and b not in urdf_pose: urdf_pose[b]=urdf_pose[a]@t
        elif b in urdf_pose and a not in urdf_pose: urdf_pose[a]=urdf_pose[b]@np.linalg.inv(t)
    if len(urdf_pose)==before: break
usd_edges=[]
for p in stage.Traverse():
    if not p.IsA(UsdPhysics.FixedJoint): continue
    j=UsdPhysics.Joint(p); a=j.GetBody0Rel().GetTargets(); b=j.GetBody1Rel().GetTargets(); assert len(a)==len(b)==1
    t0=matrix(quat(j.GetLocalRot0Attr().Get()),np.array(j.GetLocalPos0Attr().Get()));t1=matrix(quat(j.GetLocalRot1Attr().Get()),np.array(j.GetLocalPos1Attr().Get()))
    usd_edges.append((a[0].name,b[0].name,t0@np.linalg.inv(t1)))
usd_pose={ANCHOR:np.eye(4)}
while True:
    before=len(usd_pose)
    for a,b,t in usd_edges:
        if a in usd_pose and b not in usd_pose: usd_pose[b]=usd_pose[a]@t
        elif b in usd_pose and a not in usd_pose: usd_pose[a]=usd_pose[b]@np.linalg.inv(t)
    if len(usd_pose)==before: break
def urdf_part(name,T):
    inertial=links[name].find('inertial')
    if inertial is None:return None
    origin=inertial.find('origin'); a=origin.attrib if origin is not None else {}; F=T@matrix(Rotation.from_euler('xyz',vec(a.get('rpy','0 0 0'))).as_matrix(),vec(a.get('xyz','0 0 0')))
    mass=float(inertial.find('mass').get('value'));q=inertial.find('inertia').attrib
    I=np.array([[float(q['ixx']),float(q['ixy']),float(q['ixz'])],[float(q['ixy']),float(q['iyy']),float(q['iyz'])],[float(q['ixz']),float(q['iyz']),float(q['izz'])]])
    return {'mass':mass,'center':F[:3,3],'inertia':F[:3,:3]@I@F[:3,:3].T}
def usd_part(name,T):
    p=stage.GetPrimAtPath('/g1_29dof_with_hand_rev_1_0/'+name);m=UsdPhysics.MassAPI(p)
    attrs=[m.GetMassAttr(),m.GetCenterOfMassAttr(),m.GetDiagonalInertiaAttr(),m.GetPrincipalAxesAttr()];assert all(a.HasAuthoredValueOpinion() for a in attrs)
    mass=float(attrs[0].Get());c=T[:3,:3]@np.array(attrs[1].Get())+T[:3,3]; R=T[:3,:3]@quat(attrs[3].Get());I=R@np.diag(np.array(attrs[2].Get()))@R.T
    return {'mass':mass,'center':c,'inertia':I}
up={n:urdf_part(n,t) for n,t in urdf_pose.items()};dp={n:usd_part(n,t) for n,t in usd_pose.items()}
def serial(p): return {n:None if v is None else {'mass_kg':v['mass'],'center_in_anchor_m':v['center'].tolist(),'inertia_at_center_anchor_kg_m2':v['inertia'].tolist()} for n,v in p.items()}
u=aggregate([v for v in up.values() if v is not None]); d=aggregate(list(dp.values()))
out={'anchor':ANCHOR,'urdf_sha256':sha(URDF),'usd_sha256':sha(USD),'script_sha256':sha(Path(__file__)),'urdf_components':serial(up),'usd_components':serial(dp),'urdf_declared_aggregate':u,'usd_declared_aggregate':d,'center_difference_m':(np.array(u['center_in_anchor_m'])-d['center_in_anchor_m']).tolist(),'limitations':['URDF links lacking inertial are excluded, not established massless.','USD mass tensor uses authored diagonal inertia and principal axes, without runtime PhysX verification.','Asset composition, collision, actual mass properties and articulation runtime are not proven equivalent.']}
with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False)

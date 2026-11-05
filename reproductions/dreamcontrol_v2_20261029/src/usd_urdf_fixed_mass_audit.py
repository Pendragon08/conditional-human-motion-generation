"""Recompute one fixed waist mass cluster from raw Sdf/XML and elementary formulas."""
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from pxr import Sdf

BASE=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')
USD=BASE/'g1_27dof_with_hands_min_collisions_flat_white.usd'; URDF=BASE/'g1_27dof.urdf'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rotation(q):
    q=np.asarray(q,dtype=float);q=q/np.linalg.norm(q);w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
assert np.max(np.abs(rotation([math.sqrt(.5),0,0,math.sqrt(.5)])@np.array([1,0,0])-np.array([0,1,0]))) < 1e-12
report=Path(sys.argv[1]); r=json.loads(report.read_text()); assert sha(USD)==r['usd_sha256'] and sha(URDF)==r['urdf_sha256']
root=ET.parse(URDF).getroot(); L={x.get('name'):x for x in root.findall('link')};J={x.get('name'):x for x in root.findall('joint')};layer=Sdf.Layer.FindOrOpen(str(USD)); prefix='/g1_29dof_with_hand_rev_1_0/'
def origin(j):
    o=J[j].find('origin'); a=o.attrib if o is not None else {}; assert np.linalg.norm(np.fromstring(a.get('rpy','0 0 0'),sep=' '))==0
    return np.fromstring(a.get('xyz','0 0 0'),sep=' ')
uroll=origin('waist_roll_joint'); utorso=uroll+origin('waist_pitch_joint')
offsets={'waist_yaw_link':np.zeros(3),'waist_roll_link':uroll,'torso_link':utorso}
for n in ('logo','head','waist_support'):
    assert J[n+'_joint'].find('parent').get('link')=='torso_link'; offsets[n+'_link']=utorso+origin(n+'_joint')
assert {k for k,v in r['urdf_components'].items() if v is not None} == set(offsets)
def raw(name,attr): return layer.GetPrimAtPath(prefix+name).attributes[attr].default
def usd_joint(n):
    for i in (0,1):
        q=raw('joints/'+n,'physics:localRot'+str(i));assert abs(float(q.GetReal())-1)<1e-12 and np.linalg.norm(np.array(q.GetImaginary()))<1e-12
    return np.array(raw('joints/'+n,'physics:localPos0'),dtype=float)-np.array(raw('joints/'+n,'physics:localPos1'),dtype=float)
droll=usd_joint('waist_roll_joint'); dtorso=droll+usd_joint('waist_pitch_joint');doff={'waist_yaw_link':np.zeros(3),'waist_roll_link':droll,'torso_link':dtorso}
assert set(doff)==set(r['usd_components'])
parts={}; worst=0.0
for n,o in offsets.items():
    i=L[n].find('inertial'); a=i.find('origin').attrib;assert np.linalg.norm(np.fromstring(a.get('rpy','0 0 0'),sep=' '))==0
    c=o+np.fromstring(a.get('xyz','0 0 0'),sep=' '); mass=float(i.find('mass').get('value'));t=i.find('inertia').attrib
    I=np.array([[float(t['ixx']),float(t['ixy']),float(t['ixz'])],[float(t['ixy']),float(t['iyy']),float(t['iyz'])],[float(t['ixz']),float(t['iyz']),float(t['izz'])]])
    parts[n]=(mass,c,I)
for n,o in doff.items():
    mass=float(raw(n,'physics:mass'));c=o+np.array(raw(n,'physics:centerOfMass'),dtype=float); diag=np.array(raw(n,'physics:diagonalInertia'),dtype=float);q=raw(n,'physics:principalAxes');R=rotation([q.GetReal(),*q.GetImaginary()]);I=R@np.diag(diag)@R.T
    parts['USD:'+n]=(mass,c,I)
def check(kind, names):
    global worst
    pp=[parts[n] for n in names]; mass=sum(p[0] for p in pp); center=sum((m*c for m,c,I in pp),np.zeros(3))/mass
    I=sum((I+m*((np.dot(c-center,c-center))*np.eye(3)-np.outer(c-center,c-center)) for m,c,I in pp),np.zeros((3,3)))
    expected=r[kind+'_declared_aggregate'];err=max(abs(mass-expected['mass_kg']),np.max(np.abs(center-expected['center_in_anchor_m'])),np.max(np.abs(I-expected['inertia_at_center_anchor_kg_m2'])));worst=max(worst,float(err));print(kind, 'max component difference:', err);assert err<1e-9
check('urdf',list(offsets));check('usd',['USD:'+n for n in doff])
out={'status':'passed','report_sha256':sha(report),'urdf_sha256':sha(URDF),'usd_sha256':sha(USD),'max_mass_center_inertia_numeric_difference':worst,'scope':'One explicitly enumerated fixed waist group; independent raw Sdf/XML and explicit quaternion and parallel-axis formulas. Not PhysX validation.'}
with Path(sys.argv[2]).open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)

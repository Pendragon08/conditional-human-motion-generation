"""Raw-layer and manual Rodrigues recheck of 27 USD/URDF joint transforms."""
import hashlib,json,math,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from pxr import Sdf
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');BASE=ROOT/'isaac_usd_urdf_joint_frames_20261030';OUT=Path(__file__).resolve().parent
TRAIN=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training');USD=TRAIN/'HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof_with_hands_min_collisions_flat_white.usd';URDF=TRAIN/'HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def Rq(q):
    v=np.array(q.GetImaginary(),float);w=float(q.GetReal());n=math.sqrt(w*w+v@v);w/=n;v/=n
    x,y,z=v;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]]);return np.eye(3)+2*w*K+2*K@K
def axisrot(axis,angle):
    x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]]);return np.eye(3)+math.sin(angle)*K+(1-math.cos(angle))*K@K
def main():
    r=json.loads((BASE/'report.json').read_text());assert sha(USD)==r['usd_sha256'] and sha(URDF)==r['urdf_sha256'] and sha(BASE/'audit.py')==r['script_sha256']
    layer=Sdf.Layer.FindOrOpen(str(USD));xml={x.get('name'):x for x in ET.parse(URDF).getroot().findall('joint')};diffs=[];shoulders=[]
    for row in r['joints']:
        j=xml[row['joint']];sp=layer.GetPrimAtPath('/g1_29dof_with_hand_rev_1_0/joints/'+row['joint']);attrs={a.name:a.default for a in sp.attributes};o=j.find('origin');xyz=np.array(list(map(float,o.get('xyz','0 0 0').split())));roll,pitch,yaw=map(float,o.get('rpy','0 0 0').split())
        rx=axisrot(np.array([1,0,0]),roll);ry=axisrot(np.array([0,1,0]),pitch);rz=axisrot(np.array([0,0,1]),yaw);U=rz@ry@rx
        node=j.find('axis');ua=np.array(list(map(float,node.get('xyz','1 0 0').split())));ua/=np.linalg.norm(ua);da=np.eye(3)['XYZ'.index(str(attrs['physics:axis']))]
        q0=Rq(attrs['physics:localRot0']);q1=Rq(attrs['physics:localRot1']);p0=np.array(attrs['physics:localPos0'],float);p1=np.array(attrs['physics:localPos1'],float)
        for sample in row['samples']:
            a=sample['angle_rad'];ur=U@axisrot(ua,a);dr=q0@axisrot(da,a)@q1.T;dt=p0-dr@p1;position=float(np.linalg.norm(xyz-dt))
            rot_err=np.linalg.norm(ur-dr,ord='fro')/math.sqrt(2);assert abs(position-sample['position_difference_m'])<1e-12 and abs(rot_err-sample['rotation_difference_rad'])<1e-9
            diffs.append(position)
        if 'shoulder_pitch' in row['joint']:
            assert np.linalg.norm(p1)<1e-12;shoulders.append({'joint':row['joint'],'urdf_xyz_m':xyz.tolist(),'usd_pos0_m':p0.tolist(),'usd_minus_urdf_m':(p0-xyz).tolist()})
    assert len(diffs)==81 and len(shoulders)==2
    out={'primary_report_sha256':sha(BASE/'report.json'),'script_sha256':sha(Path(__file__)),'samples_checked':len(diffs),'max_position_difference_m':max(diffs),'shoulder_joint_origins':shoulders,
         'limitations':'Raw authored local frames; no actual USD runtime rigid-body frames, scaling, PhysX, dynamics, collision or training acceptance.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out))
if __name__=='__main__':main()

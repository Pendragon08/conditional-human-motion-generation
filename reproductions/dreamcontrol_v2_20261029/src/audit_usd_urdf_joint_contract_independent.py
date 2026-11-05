"""Raw Sdf and XML recheck of authored joint limits and parent-frame axes."""
import hashlib,json,math,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from pxr import Sdf
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');BASE=ROOT/'isaac_usd_urdf_joint_contract_20261030';OUT=Path(__file__).resolve().parent
TRAIN=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training')
USD=TRAIN/'HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof_with_hands_min_collisions_flat_white.usd';URDF=TRAIN/'HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    report=json.loads((BASE/'report.json').read_text());assert sha(USD)==report['usd_sha256'] and sha(URDF)==report['urdf_sha256'] and sha(BASE/'audit.py')==report['script_sha256']
    layer=Sdf.Layer.FindOrOpen(str(USD));xml={j.get('name'):j for j in ET.parse(URDF).getroot().findall('joint')};errors=[]
    for row in report['joints']:
        name=row['joint'];joint=xml[name];spec=layer.GetPrimAtPath('/g1_29dof_with_hand_rev_1_0/joints/'+name);assert spec and spec.typeName=='PhysicsRevoluteJoint'
        attrs={x.name:x.default for x in spec.attributes};rels={x.name:x.targetPathList.GetAddedOrExplicitItems() for x in spec.relationships}
        assert str(rels['physics:body0'][0].name)==joint.find('parent').get('link')==row['urdf_parent']
        assert str(rels['physics:body1'][0].name)==joint.find('child').get('link')==row['urdf_child']
        low=float(attrs['physics:lowerLimit'])*math.pi/180;high=float(attrs['physics:upperLimit'])*math.pi/180
        assert abs(low-row['usd_limits_rad'][0])<1e-12 and abs(high-row['usd_limits_rad'][1])<1e-12
        limit=joint.find('limit');assert float(limit.get('lower'))==row['urdf_limits_rad'][0] and float(limit.get('upper'))==row['urdf_limits_rad'][1]
        quat=attrs['physics:localRot0'];w=float(quat.GetReal());xyz=np.array(quat.GetImaginary(),dtype=float);norm=math.sqrt(w*w+float(xyz@xyz));w/=norm;xyz/=norm;axis=np.eye(3)['XYZ'.index(str(attrs['physics:axis']))]
        # Unit quaternion rotation formula, independently of scipy Rotation.
        usdaxis=axis+2*w*np.cross(xyz,axis)+2*np.cross(xyz,np.cross(xyz,axis))
        origin=joint.find('origin');roll,pitch,yaw=map(float,(origin.get('rpy','0 0 0') if origin is not None else '0 0 0').split());cr,sr=math.cos(roll),math.sin(roll);cp,sp=math.cos(pitch),math.sin(pitch);cy,sy=math.cos(yaw),math.sin(yaw)
        R=np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
        node=joint.find('axis');urdfaxis=np.array(list(map(float,(node.get('xyz') if node is not None else '1 0 0').split())));urdfaxis/=np.linalg.norm(urdfaxis)
        error=float(np.linalg.norm(R@urdfaxis-usdaxis));assert abs(error-row['axis_parent_frame_difference'])<1e-12;errors.append(error)
    out={'primary_report_sha256':sha(BASE/'report.json'),'script_sha256':sha(Path(__file__)),'checked_joints':len(errors),'max_parent_axis_difference':max(errors),
         'limitations':'Raw authored values only, independent quaternion and RPY formula; not full joint frame/zero, runtime soft limits, drive, contacts, PhysX or training acceptance.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out))
if __name__=='__main__':main()

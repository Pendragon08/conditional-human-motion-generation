"""Compare authored 27 joint relative transforms at three test angles."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from pxr import Usd,UsdPhysics
from scipy.spatial.transform import Rotation
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');OUT=Path(__file__).resolve().parent
TRAIN=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training')
USD=TRAIN/'HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof_with_hands_min_collisions_flat_white.usd';URDF=TRAIN/'HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
BASE=ROOT/'isaac_usd_urdf_joint_contract_20261030/report.json'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def fromquat(q):
    v=q.GetImaginary();return Rotation.from_quat([v[0],v[1],v[2],q.GetReal()]).as_matrix()
def main():
    doc=json.loads(BASE.read_text());assert sha(USD)==doc['usd_sha256'] and sha(URDF)==doc['urdf_sha256']
    xml={j.get('name'):j for j in ET.parse(URDF).getroot().findall('joint')};stage=Usd.Stage.Open(str(USD));joints={p.GetName():UsdPhysics.RevoluteJoint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)};rows=[]
    for rec in doc['joints']:
        name=rec['joint'];u=xml[name];v=joints[name];origin=u.find('origin');rpy=list(map(float,origin.get('rpy','0 0 0').split())) if origin is not None else [0]*3;xyz=list(map(float,origin.get('xyz','0 0 0').split())) if origin is not None else [0]*3
        U=Rotation.from_euler('xyz',rpy).as_matrix();axis=u.find('axis');ua=np.array(list(map(float,axis.get('xyz','1 0 0').split())));ua/=np.linalg.norm(ua)
        Q0=fromquat(v.GetLocalRot0Attr().Get());Q1=fromquat(v.GetLocalRot1Attr().Get());p0=np.array(v.GetLocalPos0Attr().Get(),dtype=float);p1=np.array(v.GetLocalPos1Attr().Get(),dtype=float);da=np.eye(3)['XYZ'.index(str(v.GetAxisAttr().Get()))]
        low,high=rec['urdf_limits_rad'];angles=[0.,low*.5,high*.5];samples=[]
        for theta in angles:
            # Both formats rotate about their joint-frame axis. Compare child-to-parent
            # transform after composing each authored local frame.
            ur=U@Rotation.from_rotvec(ua*theta).as_matrix();ut=np.array(xyz)
            dr=Q0@Rotation.from_rotvec(da*theta).as_matrix()@Q1.T;dt=p0-dr@p1
            samples.append({'angle_rad':theta,'position_difference_m':float(np.linalg.norm(ut-dt)),
                            'rotation_difference_rad':float(Rotation.from_matrix(ur.T@dr).magnitude())})
        rows.append({'joint':name,'samples':samples})
    samples=[s for r in rows for s in r['samples']]
    out={'usd_sha256':sha(USD),'urdf_sha256':sha(URDF),'joint_contract_report_sha256':sha(BASE),'script_sha256':sha(Path(__file__)),'joints':rows,
         'max_position_difference_m':max(s['position_difference_m'] for s in samples),'max_rotation_difference_rad':max(s['rotation_difference_rad'] for s in samples),
         'limitations':'Analytical local rigid transform composition at three angles per joint from authored USD/URDF; not actual PhysX runtime, mass, actuator, collision filtering, dynamic equivalence or data acceptance.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({k:out[k] for k in ('max_position_difference_m','max_rotation_difference_rad')}))
if __name__=='__main__':main()

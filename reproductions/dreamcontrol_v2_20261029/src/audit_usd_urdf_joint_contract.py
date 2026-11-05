"""Compare 27 named USD/URDF joint metadata; not a PhysX dynamics test."""
import hashlib,json,math,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from pxr import Usd,UsdPhysics
from scipy.spatial.transform import Rotation
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');OUT=Path(__file__).resolve().parent
TRAIN=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training')
USD=TRAIN/'HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof_with_hands_min_collisions_flat_white.usd'
URDF=TRAIN/'HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
PREV=ROOT/'isaac_g1_usd_static_v2_20261030/report.json';PYROKI=ROOT/'pyroki_grab_s5_cup_windowed_20261030/report.json'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def vec(s,default):return np.array([float(x) for x in (s if s is not None else default).split()])
def main():
    prev=json.loads(PREV.read_text());pr=json.loads(PYROKI.read_text());assert sha(USD)==prev['usd_sha256'] and sha(URDF)==pr['urdf_sha256']
    order=prev['revolute_joint_names_in_usd_traversal'];body=[order[i] for i in prev['body_joint_indices_in_usd_traversal']];assert len(body)==27
    root=ET.parse(URDF).getroot();uj={j.attrib['name']:j for j in root.findall('joint')};stage=Usd.Stage.Open(str(USD));sj={p.GetName():UsdPhysics.RevoluteJoint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)};rows=[]
    for name in body:
        a=uj[name];b=sj[name];parent=a.find('parent').attrib['link'];child=a.find('child').attrib['link'];b0=b.GetBody0Rel().GetTargets();b1=b.GetBody1Rel().GetTargets()
        if len(b0)!=1 or len(b1)!=1:raise ValueError('unexpected USD body targets')
        usdparent=b0[0].name;usdchild=b1[0].name;origin=a.find('origin');rpy=vec(origin.get('rpy') if origin is not None else None,'0 0 0');uaxis=vec(a.find('axis').get('xyz') if a.find('axis') is not None else None,'1 0 0');uaxis/=np.linalg.norm(uaxis)
        uparent=Rotation.from_euler('xyz',rpy).apply(uaxis);axis=str(b.GetAxisAttr().Get());basis={'X':[1.,0,0],'Y':[0,1.,0],'Z':[0,0,1.]}[axis];rot=b.GetLocalRot0Attr().Get();im=rot.GetImaginary();quat=[im[0],im[1],im[2],rot.GetReal()];dparent=Rotation.from_quat(quat).apply(basis)
        l=a.find('limit');ul=float(l.attrib['lower']);uu=float(l.attrib['upper']);dl=math.radians(float(b.GetLowerLimitAttr().Get()));du=math.radians(float(b.GetUpperLimitAttr().Get()))
        rows.append({'joint':name,'urdf_parent':parent,'usd_parent':usdparent,'urdf_child':child,'usd_child':usdchild,
            'parents_match':parent==usdparent,'children_match':child==usdchild,'axis_parent_frame_difference':float(np.linalg.norm(uparent-dparent)),
            'urdf_limits_rad':[ul,uu],'usd_limits_rad':[dl,du],'maximum_numerical_limit_difference_rad':max(abs(ul-dl),abs(uu-du))})
    out={'usd_sha256':sha(USD),'urdf_sha256':sha(URDF),'usd_inventory_sha256':sha(PREV),'pyroki_report_sha256':sha(PYROKI),'script_sha256':sha(Path(__file__)),
        'joint_count':len(rows),'parent_mismatch_count':sum(not r['parents_match'] for r in rows),'child_mismatch_count':sum(not r['children_match'] for r in rows),
        'max_axis_parent_difference':max(r['axis_parent_frame_difference'] for r in rows),'max_numeric_limit_difference_rad':max(r['maximum_numerical_limit_difference_rad'] for r in rows),'joints':rows,
        'limitations':'Compares authored USD/URDF metadata, no runtime articulation order, initialization, offset/zero-convention equivalence, soft limit, collision, actuators or dynamic simulation. Joint limit numerical agreement alone does not establish full kinematic or physical equivalence.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({k:out[k] for k in ('joint_count','parent_mismatch_count','child_mismatch_count','max_axis_parent_difference','max_numeric_limit_difference_rad')}))
if __name__=='__main__':main()

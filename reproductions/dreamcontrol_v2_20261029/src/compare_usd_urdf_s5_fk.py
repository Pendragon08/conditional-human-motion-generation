"""Compare 29-body-link FK reconstructed from authored USD and URDF, no simulator."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from pxr import Usd,UsdPhysics
from scipy.spatial.transform import Rotation
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');OUT=Path(__file__).resolve().parent
TRAIN=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training');USD=TRAIN/'HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof_with_hands_min_collisions_flat_white.usd';URDF=TRAIN/'HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
sys.path.insert(0,str(ROOT/'foot_contact_fit_20261030'));from g1_fk import URDFFK
sys.path.insert(0,str(ROOT/'grab_s3_object_wrist_audit_v2_20261030'));from grab_s3_object_wrist_audit_v2 import chosen,load
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def qmat(q):
    v=q.GetImaginary();return Rotation.from_quat([v[0],v[1],v[2],q.GetReal()]).as_matrix()
def main():
    contract=ROOT/'isaac_usd_urdf_joint_contract_20261030/report.json';doc=json.loads(contract.read_text());assert sha(USD)==doc['usd_sha256'] and sha(URDF)==doc['urdf_sha256']
    run=ROOT/'pyroki_grab_s5_cup_windowed_20261030';r=json.loads((run/'report.json').read_text());assert sha(run/'windows.jsonl')==r['rows_sha256']
    rows=[json.loads(x) for x in (run/'windows.jsonl').read_text().splitlines()];ws=[load(run/x['artifact'],x['artifact_sha256']) for x in rows];frames,roots,q=chosen(ws);names=ws[0]['joint_names'].tolist();assert frames.tolist()==list(range(309,427))
    stage=Usd.Stage.Open(str(USD));byname={p.GetName():p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}
    bodies=names+['waist_roll_joint','waist_pitch_joint'];assert len(bodies)==len(set(bodies))==29
    joints=[]
    for name in bodies:
        p=byname[name];j=UsdPhysics.Joint(p);a=j.GetBody0Rel().GetTargets();b=j.GetBody1Rel().GetTargets();assert len(a)==len(b)==1
        p0=np.array(j.GetLocalPos0Attr().Get(),float);p1=np.array(j.GetLocalPos1Attr().Get(),float);R0=qmat(j.GetLocalRot0Attr().Get());R1=qmat(j.GetLocalRot1Attr().Get())
        if p.IsA(UsdPhysics.RevoluteJoint):axis=np.eye(3)['XYZ'.index(str(UsdPhysics.RevoluteJoint(p).GetAxisAttr().Get()))]
        elif p.IsA(UsdPhysics.FixedJoint):axis=None
        else:raise ValueError('unsupported joint')
        joints.append((name,a[0].name,b[0].name,p0,p1,R0,R1,axis))
    links=['pelvis']+[x[2] for x in joints];assert len(links)==len(set(links))==30
    T=len(frames);rootR=Rotation.from_quat(roots[:,[1,2,3,0]]).as_matrix();positions={'pelvis':roots[:,4:].astype(float)};rots={'pelvis':rootR};pending=joints.copy();lookup={n:i for i,n in enumerate(names)}
    while pending:
        rest=[]
        for name,parent,child,p0,p1,R0,R1,axis in pending:
            if parent not in positions:rest.append((name,parent,child,p0,p1,R0,R1,axis));continue
            ar=np.broadcast_to(np.eye(3),(T,3,3)) if axis is None else Rotation.from_rotvec(q[:,lookup[name],None]*axis).as_matrix()
            relative=R0@ar@R1.T;offset=p0-np.einsum('tij,j->ti',relative,p1)
            positions[child]=positions[parent]+np.einsum('tij,tj->ti',rots[parent],offset);rots[child]=rots[parent]@relative
        if len(rest)==len(pending):raise ValueError('USD kinematic graph disconnected or cyclic')
        pending=rest
    a=np.stack([positions[x] for x in links],axis=1);ar=np.stack([rots[x] for x in links],axis=1)
    fk=URDFFK(URDF);b,br=fk.forward(roots[:,4:],roots[:,[1,2,3,0]],q,names,links=links,return_rotations=True)
    delta=a-b;dist=np.linalg.norm(delta,axis=2);relative=np.einsum('tlji,tljk->tlik',br,ar);rot_err=Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude().reshape(T,len(links))
    # The previously established shoulder offset predicts a common shift for each arm descendant.
    torso=br[:,links.index('torso_link')];pred=np.einsum('tij,j->ti',torso,np.array([0.,0.,.01]));arm=[i for i,n in enumerate(links) if n.startswith(('left_shoulder_','left_elbow','left_wrist_','right_shoulder_','right_elbow','right_wrist_'))]
    predicted_error=float(np.max(np.linalg.norm(delta[:,arm,:]-pred[:,None,:],axis=2)));other=[i for i in range(len(links)) if i not in arm];other_error=float(np.max(dist[:,other]))
    output={'usd_sha256':sha(USD),'urdf_sha256':sha(URDF),'source_window_report_sha256':sha(run/'report.json'),'window_rows_sha256':sha(run/'windows.jsonl'),'contract_report_sha256':sha(contract),'script_sha256':sha(Path(__file__)),
       'frames':len(frames),'frame_first_last':[int(frames[0]),int(frames[-1])],'links':len(links),'max_position_difference_m':float(dist.max()),'max_rotation_difference_rad':float(rot_err.max()),
       'arm_links':len(arm),'arm_offset_prediction_max_error_m':predicted_error,'other_links_max_difference_m':other_error,
       'per_link':[{'link':name,'mean_position_difference_m':float(dist[:,i].mean()),'max_position_difference_m':float(dist[:,i].max())} for i,name in enumerate(links)],
       'limitations':'Deterministic offline FK built from authored local joint frames and same chosen S5 joint/root samples; no actual Isaac stage articulation or PhysX. Only matching robot body link origins, not visual/hand mesh, contacts, forces, model truth, manipulation success or training acceptance.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps({k:output[k] for k in ('frames','links','max_position_difference_m','max_rotation_difference_rad','arm_offset_prediction_max_error_m','other_links_max_difference_m')}))
if __name__=='__main__':main()

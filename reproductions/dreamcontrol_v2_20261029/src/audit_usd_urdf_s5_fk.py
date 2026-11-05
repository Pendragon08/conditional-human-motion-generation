"""Independent raw-layer/XML FK at three S5 frames, using homogeneous matrices."""
import hashlib,json,math,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from pxr import Sdf
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');BASE=ROOT/'usd_urdf_s5_fk_20261030';RUN=ROOT/'pyroki_grab_s5_cup_windowed_20261030';OUT=Path(__file__).resolve().parent
TRAIN=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training');USD=TRAIN/'HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof_with_hands_min_collisions_flat_white.usd';URDF=TRAIN/'HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def axisrot(axis,a):
    axis=np.asarray(axis,float);axis/=np.linalg.norm(axis);x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(a)*K+(1-math.cos(a))*K@K
def qrot(w,v):
    v=np.asarray(v,float);n=math.sqrt(w*w+v@v);w/=n;v/=n;x,y,z=v;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+2*w*K+2*K@K
def transform(R,t):
    out=np.eye(4);out[:3,:3]=R;out[:3,3]=t;return out
def main():
    d=json.loads((BASE/'report.json').read_text());assert sha(USD)==d['usd_sha256'] and sha(URDF)==d['urdf_sha256'] and sha(BASE/'compare.py')==d['script_sha256'] and sha(RUN/'windows.jsonl')==d['window_rows_sha256']
    selected={};priority=[1,2,3,4,3,2,1];names=None
    for row in map(json.loads,(RUN/'windows.jsonl').read_text().splitlines()):
        path=RUN/row['artifact'];assert sha(path)==row['artifact_sha256']
        with np.load(path,allow_pickle=False) as data:
            if names is None:names=data['joint_names'].tolist()
            for i,f in enumerate(data['frames']):
                f=int(f)
                if f not in selected or priority[i]>selected[f][0]:selected[f]=(priority[i],data['root_wxyz_xyz'][i].copy(),data['joint_pos'][i].copy())
    layer=Sdf.Layer.FindOrOpen(str(USD));xml={j.get('name'):j for j in ET.parse(URDF).getroot().findall('joint')};allnames=names+['waist_roll_joint','waist_pitch_joint'];specs=[]
    for name in allnames:
        e=xml[name];sp=layer.GetPrimAtPath('/g1_29dof_with_hand_rev_1_0/joints/'+name);attrs={a.name:a.default for a in sp.attributes};rels={a.name:a.targetPathList.GetAddedOrExplicitItems() for a in sp.relationships}
        p=rels['physics:body0'][0].name;c=rels['physics:body1'][0].name;assert p==e.find('parent').get('link') and c==e.find('child').get('link')
        uq0=attrs['physics:localRot0'];uq1=attrs['physics:localRot1'];U0=transform(qrot(float(uq0.GetReal()),uq0.GetImaginary()),np.array(attrs['physics:localPos0'],float));U1=transform(qrot(float(uq1.GetReal()),uq1.GetImaginary()),np.array(attrs['physics:localPos1'],float))
        origin=e.find('origin');xyz=np.array(list(map(float,origin.get('xyz','0 0 0').split())));r,pitch,y=map(float,origin.get('rpy','0 0 0').split());R=axisrot([0,0,1],y)@axisrot([0,1,0],pitch)@axisrot([1,0,0],r);U=transform(R,xyz)
        if name in names:
            du=np.eye(3)['XYZ'.index(str(attrs['physics:axis']))];au=np.array(list(map(float,e.find('axis').get('xyz','1 0 0').split())));au/=np.linalg.norm(au)
        else:du=au=None
        specs.append((name,p,c,U0,U1,U,du,au))
    result=[]
    for frame in (309,367,426):
        _,root,q=selected[frame];rootR=qrot(float(root[0]),root[1:4]);world=transform(rootR,root[4:]);usd={'pelvis':world};urdf={'pelvis':world};pending=specs.copy();lookup=dict(zip(names,q))
        while pending:
            todo=[]
            for name,p,c,U0,U1,U,du,au in pending:
                if p not in usd:todo.append((name,p,c,U0,U1,U,du,au));continue
                angle=float(lookup[name]) if du is not None else 0.;A=transform(axisrot(du,angle),[0,0,0]) if du is not None else np.eye(4);B=transform(axisrot(au,angle),[0,0,0]) if au is not None else np.eye(4)
                usd[c]=usd[p]@U0@A@np.linalg.inv(U1);urdf[c]=urdf[p]@U@B
            if len(todo)==len(pending):raise ValueError('disconnected chain')
            pending=todo
        assert usd.keys()==urdf.keys();arm=[name for name in usd if name.startswith(('left_shoulder_','left_elbow','left_wrist_','right_shoulder_','right_elbow','right_wrist_'))]
        delta={name:usd[name][:3,3]-urdf[name][:3,3] for name in usd};armmax=max(np.linalg.norm(delta[x]) for x in arm)
        torso_err=float(np.linalg.norm(delta['torso_link']-rootR@np.array([0.,0.,-.01])));waist_err=float(np.linalg.norm(delta['waist_roll_link']-rootR@np.array([0.,0.,.009])))
        assert armmax<1e-6 and torso_err<1e-6 and waist_err<1e-6
        result.append({'frame':frame,'arm_max_position_difference_m':float(armmax),'torso_offset_m':float(np.linalg.norm(delta['torso_link'])),'waist_roll_offset_m':float(np.linalg.norm(delta['waist_roll_link'])),'torso_prediction_error_m':torso_err,'waist_prediction_error_m':waist_err})
    out={'primary_report_sha256':sha(BASE/'report.json'),'audit_script_sha256':sha(Path(__file__)),'frames':result,
         'limitations':'Raw Sdf and XML offline homogeneous FK for 3 of 118 frames; no real Isaac articulation state, collision geometry, ground truth, forces, dynamics or training acceptance.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out))
if __name__=='__main__':main()

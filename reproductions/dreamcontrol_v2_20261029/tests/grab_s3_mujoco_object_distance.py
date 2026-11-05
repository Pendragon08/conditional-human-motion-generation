"""Static G1 hand-collider vs GRAB apple convex hull; not a grasp test."""
import hashlib,json,sys,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
import trimesh,mujoco
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0,str(ROOT/'grab_s3_object_wrist_audit_v2_20261030'))
sys.path.insert(1,str(ROOT/'grab_s3_human_source_20261030'))
from grab_s3_object_wrist_audit_v2 import chosen,load
from grab_restricted_reader import read_grab_field

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    # Independent simple sanity check for signed distance between two spheres.
    check=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="a" type="sphere" size=".1" pos="0 0 0"/><geom name="b" type="sphere" size=".1" pos=".5 0 0"/></worldbody></mujoco>')
    cd=mujoco.MjData(check);mujoco.mj_forward(check,cd)
    simple=float(mujoco.mj_geomDistance(check,cd,0,1,1.,None))
    if abs(simple-.3)>1e-6:raise ValueError('MuJoCo distance sanity check failed')
    base=ROOT/'pyroki_grab_s3_windowed_20261030';early=ROOT/'pyroki_s3_overlap_anchor_20261030';tail=ROOT/'pyroki_s3_anchor_tail_20261030'
    bdoc=json.loads((base/'report.json').read_text());adoc=json.loads((early/'report.json').read_text());tdoc=json.loads((tail/'report.json').read_text())
    if any(sha(p/'windows.jsonl')!=d['rows_sha256'] for p,d in ((base,bdoc),(early,adoc),(tail,tdoc))):raise ValueError('window manifest changed')
    br=[json.loads(x) for x in (base/'windows.jsonl').read_text().splitlines()]
    ar={x['window']:x for x in map(json.loads,(early/'windows.jsonl').read_text().splitlines())}
    tr={x['window']:x for x in map(json.loads,(tail/'windows.jsonl').read_text().splitlines())}
    if len(br)!=29 or len(ar)!=5 or len(tr)!=23:raise ValueError('missing windows')
    src=ROOT/'grab_s3_human_source_20261030/human_source.npz';sr=json.loads((src.parent/'report.json').read_text())
    if any(sha(src)!=d['source_sha256'] for d in (bdoc,adoc,tdoc)):raise ValueError('human source mismatch')
    options=[[],[]]
    for i,row in enumerate(br):
        first=load(base/row['artifact'],row['artifact_sha256'])
        second=first if i==0 else load((early if i<=5 else tail)/f'window_{i:02d}.npz',(ar if i<=5 else tr)[i]['artifact_sha256'])
        if not np.array_equal(first['frames'],second['frames']) or not np.array_equal(first['joint_names'],second['joint_names']):raise ValueError('not matched')
        options[0].append(first);options[1].append(second)
    frames,r0,q0=chosen(options[0]);frames2,r1,q1=chosen(options[1]);names=options[0][0]['joint_names'].tolist()
    if not np.array_equal(frames,frames2) or not np.array_equal(frames,np.arange(264,382)):raise ValueError('timeline mismatch')
    archive=ROOT/'data/GRAB/grab__s3.zip'
    if sha(archive)!=sr['archive_sha256']:raise ValueError('archive changed')
    with zipfile.ZipFile(archive) as z:obj=read_grab_field(z,sr['source_member'],'object');contact=read_grab_field(z,sr['source_member'],'contact')
    ref=PurePosixPath(obj['object_mesh'])
    if ref.is_absolute() or '..' in ref.parts or ref.parts[:2]!=('tools','object_meshes'):raise ValueError('bad mesh path')
    meshfile=ROOT/'grab_mesh_stage_20261030'/str(ref);orig=trimesh.load(str(meshfile),process=False);hull=orig.convex_hull
    labels=np.asarray(contact['object'])[frames];orient=np.asarray(obj['params']['global_orient'])[frames];trans=np.asarray(obj['params']['transl'])[frames]
    right=(labels==22)|((labels>=41)&(labels<=55));left=(labels==21)|((labels>=26)&(labels<=40))
    if labels.shape[1]!=len(orig.vertices):raise ValueError('contact width does not match mesh')
    scene=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description/scene_full.xml')
    spec=mujoco.MjSpec.from_file(str(scene));m=spec.add_mesh(name='grab_apple_hull')
    m.uservert=np.asarray(hull.vertices,dtype=np.float64).reshape(-1).tolist()
    m.userface=np.asarray(hull.faces,dtype=np.int32).reshape(-1).tolist()
    body=spec.worldbody.add_body(name='grab_apple',mocap=True)
    body.add_geom(name='grab_apple_geom',type=mujoco.mjtGeom.mjGEOM_MESH,meshname='grab_apple_hull',contype=1,conaffinity=1)
    model=spec.compile();data=mujoco.MjData(model)
    objid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'grab_apple_geom')
    bodyid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'grab_apple')
    mid=int(model.body_mocapid[bodyid]);rid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint');raddr=int(model.jnt_qposadr[rid])
    if min(objid,bodyid,mid,rid)<0:raise ValueError('missing model body')
    jointids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
    if any(j<0 for j in jointids):raise ValueError('missing robot joints')
    hand={}
    for side in ('left','right'):
        ids=[]
        for i in range(model.ngeom):
            bn=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[i])) or ''
            if (bn.startswith(side+'_hand_') or bn.startswith(side+'_wrist_')) and (model.geom_contype[i]!=0 or model.geom_conaffinity[i]!=0):ids.append(i)
        if not ids:raise ValueError('no hand collision geometries for '+side)
        hand[side]=ids
    meshid=int(model.geom_dataid[objid]);adr=int(model.mesh_vertadr[meshid]);num=int(model.mesh_vertnum[meshid])
    compiled_vertices=np.asarray(model.mesh_vert[adr:adr+num]);hull_vertices=np.asarray(hull.vertices)
    perframe=[];max_mesh_error=0.
    for i,frame in enumerate(frames):
        for label,root,q in (('baseline',r0,q0),('anchored',r1,q1)):
            mujoco.mj_resetData(model,data);data.qpos[raddr:raddr+3]=root[i,4:];data.qpos[raddr+3:raddr+7]=root[i,:4]
            for j,value in zip(jointids,q[i]):data.qpos[model.jnt_qposadr[j]]=value
            data.mocap_pos[mid]=trans[i];quat=Rotation.from_rotvec(orient[i]).inv().as_quat();data.mocap_quat[mid]=quat[[3,0,1,2]]
            data.qvel[:]=0;mujoco.mj_forward(model,data)
            if i==0 and label=='baseline':
                actual=data.geom_xpos[objid]+compiled_vertices@data.geom_xmat[objid].reshape(3,3).T
                expected=Rotation.from_rotvec(orient[i]).inv().apply(hull_vertices)+trans[i]
                max_mesh_error=max(float(cKDTree(actual).query(expected)[0].max()),float(cKDTree(expected).query(actual)[0].max()))
                if max_mesh_error>1e-4:raise ValueError('MuJoCo object mesh transformation mismatch')
            metrics={}
            for side in ('left','right'):
                distances=[float(mujoco.mj_geomDistance(model,data,objid,g,1.,None)) for g in hand[side]]
                metrics[side+'_min_signed_distance_m']=min(distances)
            perframe.append({'frame':int(frame),'method':label,'right_hand_labeled_vertices':int(right[i].sum()),
                             'left_hand_labeled_vertices':int(left[i].sum()),**metrics})
    summary={}
    for label in ('baseline','anchored'):
        selected=[x for x in perframe if x['method']==label and x['right_hand_labeled_vertices']>0]
        dist=np.array([x['right_min_signed_distance_m'] for x in selected])
        summary[label]={'frames':len(selected),'median_signed_distance_m':float(np.median(dist)),
                        'minimum_signed_distance_m':float(dist.min()),'max_signed_distance_m':float(dist.max()),
                        'frames_nonpositive_distance':int((dist<=0).sum()),'distance_cap_m':1.}
    p=Path(__file__).with_name('frames.jsonl')
    with p.open('x',encoding='utf8') as f:
        for row in perframe:f.write(json.dumps(row,allow_nan=False)+'\n')
    out={'source_sha256':sha(src),'archive_sha256':sha(archive),'mesh_sha256':sha(meshfile),'scene_sha256':sha(scene),
         'script_sha256':sha(__file__),'frames_sha256':sha(p),'mujoco_version':mujoco.__version__,
         'synthetic_sphere_distance_m':simple,'mesh_world_alignment_max_error_m':max_mesh_error,
         'source_mesh_vertices':len(orig.vertices),'convex_hull_vertices':len(hull.vertices),'source_is_convex':bool(orig.is_convex),
         'robot_hand_collision_geoms':{side:len(v) for side,v in hand.items()},'right_hand_labeled_frames':int(np.any(right,axis=1).sum()),
         'summary':summary,'warning':'Static selected per-frame robot poses, right hand collision shapes with finger joints at default zero, apple convex hull instead of nonconvex surface, no table or dynamics. GRAB object labels are derived contact rather than measured force; source-human and robot morphology/offset differ. Collider distances do not prove grasp success, author-v2 algorithm fidelity, Isaac Lab behavior or training acceptance.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()

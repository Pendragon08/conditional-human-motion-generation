"""Conservative independent AABB lower bounds for historical hand distance queries."""
import json, sys, zipfile
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0,str(ROOT/'grab_s3_mujoco_object_20261030'))
from grab_s3_mujoco_object_distance import sha, chosen, load, read_grab_field
sys.path.insert(0,str(ROOT/'grab_s3_scene_timeline_v2_20261030'))
from grab_s3_scene_timeline import mesh_for

def bounds(model,data,g):
    typ=int(model.geom_type[g]); p=data.geom_xpos[g]; r=data.geom_xmat[g].reshape(3,3); s=model.geom_size[g]
    if typ==int(mujoco.mjtGeom.mjGEOM_MESH):
        mid=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid])
        v=p+model.mesh_vert[a:a+n]@r.T
        return v.min(axis=0),v.max(axis=0)
    if typ==int(mujoco.mjtGeom.mjGEOM_SPHERE):ext=np.repeat(s[0],3)
    elif typ==int(mujoco.mjtGeom.mjGEOM_CAPSULE):ext=s[0]+np.abs(r[:,2])*s[1]
    elif typ==int(mujoco.mjtGeom.mjGEOM_CYLINDER):ext=s[0]*np.sqrt(r[:,0]**2+r[:,1]**2)+s[1]*np.abs(r[:,2])
    elif typ==int(mujoco.mjtGeom.mjGEOM_BOX):ext=np.abs(r)@s
    else:
        radius=float(model.geom_rbound[g])
        if not np.isfinite(radius) or radius<=0:raise ValueError('unsupported collision geometry')
        ext=np.repeat(radius,3)
    return p-ext,p+ext

def main():
    base=ROOT/'pyroki_grab_s3_windowed_20261030';early=ROOT/'pyroki_s3_overlap_anchor_20261030';tail=ROOT/'pyroki_s3_anchor_tail_20261030'
    docs=[json.loads((p/'report.json').read_text()) for p in (base,early,tail)]
    for p,d in zip((base,early,tail),docs):
        if sha(p/'windows.jsonl')!=d['rows_sha256']:raise ValueError('manifest changed')
    br=[json.loads(x) for x in (base/'windows.jsonl').read_text().splitlines()]
    ar={x['window']:x for x in map(json.loads,(early/'windows.jsonl').read_text().splitlines())}
    tr={x['window']:x for x in map(json.loads,(tail/'windows.jsonl').read_text().splitlines())}
    if (len(br),len(ar),len(tr))!=(29,5,23):raise ValueError('missing windows')
    src=ROOT/'grab_s3_human_source_20261030/human_source.npz';sr=json.loads((src.parent/'report.json').read_text())
    if any(sha(src)!=d['source_sha256'] for d in docs):raise ValueError('source changed')
    options=[[],[]]
    for i,row in enumerate(br):
        first=load(base/row['artifact'],row['artifact_sha256'])
        second=first if i==0 else load((early if i<=5 else tail)/f'window_{i:02d}.npz',(ar if i<=5 else tr)[i]['artifact_sha256'])
        if not np.array_equal(first['frames'],second['frames']) or not np.array_equal(first['joint_names'],second['joint_names']):raise ValueError('window mismatch')
        options[0].append(first);options[1].append(second)
    frames,r0,q0=chosen(options[0]);frames2,r1,q1=chosen(options[1]);names=options[0][0]['joint_names'].tolist()
    if not np.array_equal(frames,frames2) or not np.array_equal(frames,np.arange(264,382)):raise ValueError('time mismatch')
    archive=ROOT/'data/GRAB/grab__s3.zip'
    if sha(archive)!=sr['archive_sha256']:raise ValueError('archive changed')
    with zipfile.ZipFile(archive) as z:obj=read_grab_field(z,sr['source_member'],'object')
    meshfile,orig,hull=mesh_for(obj['object_mesh'])
    orient=np.asarray(obj['params']['global_orient'])[frames];trans=np.asarray(obj['params']['transl'])[frames]
    scene=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description/scene_full.xml')
    spec=mujoco.MjSpec.from_file(str(scene));m=spec.add_mesh(name='grab_apple_hull')
    m.uservert=np.asarray(hull.vertices,np.float64).reshape(-1).tolist();m.userface=np.asarray(hull.faces,np.int32).reshape(-1).tolist()
    body=spec.worldbody.add_body(name='grab_apple',mocap=True)
    body.add_geom(name='grab_apple_geom',type=mujoco.mjtGeom.mjGEOM_MESH,meshname='grab_apple_hull',contype=1,conaffinity=1)
    model=spec.compile();data=mujoco.MjData(model)
    def id(kind,name):
        v=mujoco.mj_name2id(model,kind,name)
        if v<0:raise ValueError('missing '+name)
        return v
    objid=id(mujoco.mjtObj.mjOBJ_GEOM,'grab_apple_geom');mid=int(model.body_mocapid[id(mujoco.mjtObj.mjOBJ_BODY,'grab_apple')])
    rid=id(mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint');raddr=int(model.jnt_qposadr[rid]);jointids=[id(mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
    hand=[]
    for g in range(model.ngeom):
        bn=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or ''
        if bn.startswith(('right_hand_','right_wrist_')) and (model.geom_contype[g]!=0 or model.geom_conaffinity[g]!=0):hand.append(g)
    if len(hand)!=11:raise ValueError('unexpected hand geometry count')
    origrows=[json.loads(x) for x in (ROOT/'grab_s3_mujoco_object_20261030/frames.jsonl').read_text().splitlines()]
    old={(r['frame'],r['method']):r['right_min_signed_distance_m'] for r in origrows}
    rows=[];max_historical_diff=0.
    for i,frame in enumerate(frames):
        for label,root,q in (('baseline',r0,q0),('anchored',r1,q1)):
            mujoco.mj_resetData(model,data);data.qpos[raddr:raddr+3]=root[i,4:];data.qpos[raddr+3:raddr+7]=root[i,:4]
            for j,val in zip(jointids,q[i]):data.qpos[model.jnt_qposadr[j]]=val
            data.mocap_pos[mid]=trans[i];qt=Rotation.from_rotvec(orient[i]).inv().as_quat();data.mocap_quat[mid]=qt[[3,0,1,2]]
            data.qvel[:]=0;mujoco.mj_forward(model,data)
            lo,hi=bounds(model,data,objid); dists=[]; violations=[]
            for g in hand:
                glo,ghi=bounds(model,data,g)
                lower=float(np.linalg.norm(np.maximum(0.,np.maximum(lo-ghi,glo-hi))))
                d=float(mujoco.mj_geomDistance(model,data,objid,g,1.,None));dists.append(d)
                if d+1e-6<lower:
                    violations.append({'geom':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g),'type':int(model.geom_type[g]),'distance_m':d,'aabb_lower_bound_m':lower})
            minimum=min(dists); historical=float(old[(int(frame),label)])
            max_historical_diff=max(max_historical_diff,abs(minimum-historical))
            if max_historical_diff>1e-8:raise ValueError('historical query could not be reproduced')
            rows.append({'frame':int(frame),'method':label,'minimum_distance_m':minimum,'pair_violations':violations})
    rowfile=Path(__file__).with_name('frames.jsonl')
    with rowfile.open('x',encoding='utf-8') as f:
        for x in rows:f.write(json.dumps(x,allow_nan=False)+'\n')
    report={'archive_sha256':sha(archive),'mesh_sha256':sha(meshfile),'scene_sha256':sha(scene),'source_sha256':sha(src),
            'script_sha256':sha(Path(__file__)),'rows_sha256':sha(rowfile),'frames':len(frames),'right_hand_geoms':len(hand),
            'historical_reproduction_max_diff_m':max_historical_diff,'violation_rows':[x for x in rows if x['pair_violations']],
            'warning':'Conservative world AABBs establish only lower bounds. No violation does not prove distance or grasp correctness. Static poses; no force, dynamics or Isaac Lab; the geometry and source are approximations.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'rows':len(rows),'violations':len(report['violation_rows'])}),flush=True)
if __name__=='__main__':main()

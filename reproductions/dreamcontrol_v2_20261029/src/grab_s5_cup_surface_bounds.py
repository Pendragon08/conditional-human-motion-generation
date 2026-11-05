"""Triangle-surface vs convex hand bounds on six S5 frames; CPU diagnostic only."""
import hashlib,json,sys
from pathlib import Path
import mujoco,numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'grab_hand_independent_all_20261030'))
from grab_hand_independent_distance import bracket
sys.path.insert(0,str(ROOT/'grab_s3_hand_distance_bound_audit_v2_20261030'))
from grab_s3_hand_distance_bound_audit import bounds
sys.path.insert(0,str(ROOT/'grab_s3_object_wrist_audit_v2_20261030'))
from grab_s3_object_wrist_audit_v2 import chosen,load
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def geom_points(m,d,g):
    typ=int(m.geom_type[g]);p=d.geom_xpos[g];r=d.geom_xmat[g].reshape(3,3)
    if typ==int(mujoco.mjtGeom.mjGEOM_MESH):
        mid=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid]);return p+m.mesh_vert[start:start+n]@r.T
    if typ==int(mujoco.mjtGeom.mjGEOM_BOX):
        s=m.geom_size[g];verts=np.array([[a*s[0],b*s[1],c*s[2]] for a in (-1,1) for b in (-1,1) for c in (-1,1)])
        return p+verts@r.T
    raise ValueError('unsupported geometry for constructive upper bound')
def surface_bounds(world,faces,model,data,g,tol=2e-5):
    # Existing surface vertices and hand-convex vertices yield a feasible upper bound.
    upper=float(np.min(cKDTree(world).query(geom_points(model,data,g))[0]));lower=0.
    tris=world[faces];lo=tris.min(axis=1);hi=tris.max(axis=1);gl,gh=bounds(model,data,g)
    box=np.linalg.norm(np.maximum(0.,np.maximum(lo-gh,gl-hi)),axis=1);order=np.argsort(box,kind='stable')
    lower_evaluated=float('inf');evaluated=0;not_converged=0;max_gap=0.
    for pos,index in enumerate(order):
        if box[index]>upper+1e-12:break
        if upper<=tol:
            # Zero is a global lower bound; a feasible witness is already within tolerance.
            lower=0.;return {'lower_m':lower,'upper_m':upper,'triangles_evaluated':evaluated,'nonbounded_triangle_solves':not_converged,'max_triangle_gap_m':max_gap,'stopped_near_zero':True}
        b=bracket(tris[index],model,data,g,tol=tol)
        lower_evaluated=min(lower_evaluated,max(float(box[index]),b['lower_m']));upper=min(upper,b['upper_m'])
        evaluated+=1;not_converged+=b['status']!='bounded';max_gap=max(max_gap,b['gap_m'])
    else:pos=len(order)
    remaining=float(box[order[pos]]) if pos<len(order) else float('inf')
    lower=min(lower_evaluated,remaining)
    if not np.isfinite(lower):lower=upper
    if lower>upper+1e-8:raise ValueError('lower exceeds feasible upper')
    return {'lower_m':float(max(0,lower)),'upper_m':float(upper),'triangles_evaluated':evaluated,'nonbounded_triangle_solves':not_converged,'max_triangle_gap_m':max_gap,'stopped_near_zero':False}
def main():
    prev=ROOT/'grab_s5_hand_independent_20261030';run=ROOT/'pyroki_grab_s5_cup_windowed_20261030';src=ROOT/'grab_s5_cup_human_source_20261030'
    pd=json.loads((prev/'report.json').read_text());sr=json.loads((src/'report.json').read_text());rr=json.loads((run/'report.json').read_text())
    if sha(prev/'frames.jsonl')!=pd['frame_results_sha256'] or sha(src/'human_source.npz')!=pd['source_sha256'] or sha(run/'windows.jsonl')!=pd['rows_source_sha256']:raise ValueError('source hashes changed')
    rows=[json.loads(x) for x in (run/'windows.jsonl').read_text().splitlines()];ws=[load(run/x['artifact'],x['artifact_sha256']) for x in rows];frames,roots,angles=chosen(ws);names=ws[0]['joint_names'].tolist()
    wanted=[367,368,369,370,371,372];indices=np.searchsorted(frames,wanted);assert np.array_equal(frames[indices],wanted)
    with np.load(src/'human_source.npz',allow_pickle=False) as f:
        ind=np.searchsorted(f['input_frame_indices'],wanted);assert np.array_equal(f['input_frame_indices'][ind],wanted)
        orient=f['object_orient_rotvec'][ind];trans=f['object_transl'][ind]
    meshpath=ROOT/'grab_mesh_stage_20261030'/sr['object_mesh'];assert sha(meshpath)==pd['mesh_sha256'];mesh=trimesh.load(str(meshpath),process=False)
    vertices=np.asarray(mesh.vertices,dtype=float);faces=np.asarray(mesh.faces,dtype=int)
    if len(vertices)!=20002 or len(faces)!=40000 or not np.array_equal(np.unique(faces),np.arange(len(vertices))):raise ValueError('unexpected source mesh or unused vertices')
    scene=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description/scene_full.xml');assert sha(scene)==pd['scene_sha256']
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    def mid(kind,name):
        v=mujoco.mj_name2id(model,kind,name)
        if v<0:raise ValueError('missing '+name)
        return v
    addr=int(model.jnt_qposadr[mid(mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint')]);joints=[mid(mujoco.mjtObj.mjOBJ_JOINT,n) for n in names];hand=[]
    for g in range(model.ngeom):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or ''
        if name.startswith(('right_hand_','right_wrist_')) and (model.geom_contype[g] or model.geom_conaffinity[g]):hand.append(g)
    assert len(hand)==11
    simple=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="s" type="sphere" size=".1" pos=".5 0 0"/></worldbody></mujoco>');sd=mujoco.MjData(simple);mujoco.mj_forward(simple,sd)
    triangle=np.array([[0.,-1.,-1.],[0.,1.,-1.],[0.,0.,1.]]);test=bracket(triangle,simple,sd,0)
    if not(test['lower_m']-1e-8<=.4<=test['upper_m']+1e-8 and test['gap_m']<=2e-5):raise ValueError('triangle/sphere synthetic failed')
    allrows=[]
    with (OUT/'pairs.jsonl').open('x',encoding='utf8') as f:
        for k,frame in enumerate(wanted):
            i=indices[k];world=Rotation.from_rotvec(orient[k]).inv().apply(vertices)+trans[k]
            mujoco.mj_resetData(model,data);data.qpos[addr:addr+3]=roots[i,4:];data.qpos[addr+3:addr+7]=roots[i,:4]
            for j,value in zip(joints,angles[i]):data.qpos[model.jnt_qposadr[j]]=value
            mujoco.mj_forward(model,data)
            for g in hand:
                b=surface_bounds(world,faces,model,data,g)
                record={'frame':frame,'geom_id':g,'body':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])),**b}
                f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();allrows.append(record)
                print(json.dumps({'frame':frame,'geom':g,'tested':b['triangles_evaluated']}),flush=True)
    summaries=[]
    for frame in wanted:
        a=[r for r in allrows if r['frame']==frame]
        summaries.append({'frame':frame,'lower_m':min(x['lower_m'] for x in a),'upper_m':min(x['upper_m'] for x in a),
            'triangles_evaluated':sum(x['triangles_evaluated'] for x in a),'nonbounded_triangle_solves':sum(x['nonbounded_triangle_solves'] for x in a)})
    out={'source_report_sha256':sha(src/'report.json'),'source_sha256':sha(src/'human_source.npz'),'windows_sha256':sha(run/'windows.jsonl'),'convex_report_sha256':sha(prev/'report.json'),
         'mesh_sha256':sha(meshpath),'scene_sha256':sha(scene),'script_sha256':sha(Path(__file__)),'pair_rows_sha256':sha(OUT/'pairs.jsonl'),
         'surface_faces':len(faces),'source_is_watertight':bool(mesh.is_watertight),'synthetic_triangle_sphere_bound':test,'frames':summaries,
         'limitations':'Only six selected frames, static original cup triangle surface versus convex hand colliders, same assumed FK and object pose; unsigned distance does not establish volume overlap, force, contact or grasp. Support brackets and AABB pruning have finite tolerances, and unconverged triangle bounds are explicitly counted. Not training acceptance.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()

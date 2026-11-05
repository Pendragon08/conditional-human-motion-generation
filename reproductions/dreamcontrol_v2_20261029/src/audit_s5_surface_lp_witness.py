"""Validate six-frame records and independently find four triangle/convex intersections."""
import hashlib,json,sys
from pathlib import Path
import mujoco,numpy as np,trimesh
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'grab_s3_hand_distance_bound_audit_v2_20261030'))
from grab_s3_hand_distance_bound_audit import bounds
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    primary=ROOT/'grab_s5_cup_surface_bounds_20261030';source=ROOT/'grab_s5_cup_human_source_20261030';run=ROOT/'pyroki_grab_s5_cup_windowed_20261030';cv=ROOT/'grab_s5_hand_independent_20261030'
    report=json.loads((primary/'report.json').read_text());sr=json.loads((source/'report.json').read_text());rows=[json.loads(x) for x in (primary/'pairs.jsonl').read_text().splitlines()]
    for path,field in [(primary/'run.py','script_sha256'),(primary/'pairs.jsonl','pair_rows_sha256'),(source/'human_source.npz','source_sha256'),(source/'report.json','source_report_sha256'),(run/'windows.jsonl','windows_sha256'),(cv/'report.json','convex_report_sha256')]:assert sha(path)==report[field]
    assert len(rows)==66 and len({(x['frame'],x['geom_id']) for x in rows})==66 and all(0<=x['lower_m']<=x['upper_m']+1e-8 for x in rows)
    assert {x['frame'] for x in rows}==set(range(367,373))
    convex={x['frame']:x for x in map(json.loads,(cv/'frames.jsonl').read_text().splitlines())};comparisons=[]
    for frame in range(367,373):
        subset=[x for x in rows if x['frame']==frame];s=next(x for x in report['frames'] if x['frame']==frame)
        assert abs(s['lower_m']-min(x['lower_m'] for x in subset))<1e-12 and abs(s['upper_m']-min(x['upper_m'] for x in subset))<1e-12
        assert s['triangles_evaluated']==sum(x['triangles_evaluated'] for x in subset) and s['nonbounded_triangle_solves']==sum(x['nonbounded_triangle_solves'] for x in subset)
        assert convex[frame]['lower_m']<=s['upper_m']+1e-7
        comparisons.append({'frame':frame,'convex_lower_m':convex[frame]['lower_m'],'convex_upper_m':convex[frame]['upper_m'],'surface_lower_m':s['lower_m'],'surface_upper_m':s['upper_m']})
    # Independent maximum-priority window selection; ties retain first occurrence.
    selected={};priority=(1,2,3,4,3,2,1);names=None
    for row in map(json.loads,(run/'windows.jsonl').read_text().splitlines()):
        path=run/row['artifact'];assert sha(path)==row['artifact_sha256']
        with np.load(path,allow_pickle=False) as d:
            if names is None:names=d['joint_names'].tolist()
            for j,frame in enumerate(d['frames']):
                frame=int(frame)
                if frame not in selected or priority[j]>selected[frame][0]:selected[frame]=(priority[j],d['root_wxyz_xyz'][j].copy(),d['joint_pos'][j].copy())
    with np.load(source/'human_source.npz',allow_pickle=False) as d:
        idx=np.searchsorted(d['input_frame_indices'],np.arange(368,372));assert np.array_equal(d['input_frame_indices'][idx],np.arange(368,372))
        orient=d['object_orient_rotvec'][idx];trans=d['object_transl'][idx]
    meshpath=ROOT/'grab_mesh_stage_20261030'/sr['object_mesh'];assert sha(meshpath)==report['mesh_sha256'];mesh=trimesh.load(str(meshpath),process=False)
    verts=np.asarray(mesh.vertices,dtype=float);faces=np.asarray(mesh.faces,dtype=int)
    scene=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description/scene_full.xml');assert sha(scene)==report['scene_sha256']
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    def mid(kind,name):
        v=mujoco.mj_name2id(model,kind,name)
        if v<0:raise ValueError('missing '+name)
        return v
    addr=int(model.jnt_qposadr[mid(mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint')]);joints=[mid(mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
    witnesses=[]
    for k,frame in enumerate(range(368,372)):
        _,root,q=selected[frame];mujoco.mj_resetData(model,data);data.qpos[addr:addr+3]=root[4:];data.qpos[addr+3:addr+7]=root[:4]
        for j,v in zip(joints,q):data.qpos[model.jnt_qposadr[j]]=v
        mujoco.mj_forward(model,data);geom=99;assert (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[geom])) or '')=='right_hand_index_0_link'
        mi=int(model.geom_dataid[geom]);ad=int(model.mesh_vertadr[mi]);n=int(model.mesh_vertnum[mi]);points=np.asarray(model.mesh_vert[ad:ad+n],dtype=float)
        hull=ConvexHull(points);gl,gh=bounds(model,data,geom)
        world=verts@Rotation.from_rotvec(orient[k]).as_matrix()+trans[k];triangles=world[faces];lo=triangles.min(axis=1);hi=triangles.max(axis=1)
        mask=np.all((lo<=gh+1e-9)&(hi>=gl-1e-9),axis=1);candidates=np.where(mask)[0];center=data.geom_xpos[geom];candidates=candidates[np.argsort(np.linalg.norm(triangles[candidates].mean(axis=1)-center,axis=1))]
        R=data.geom_xmat[geom].reshape(3,3);p=data.geom_xpos[geom];tested=0;found=None
        for face in candidates:
            # Map triangle into the local mesh frame; find nonnegative barycentric weights inside every hull halfspace.
            tri=(triangles[face]-p)@R;A=hull.equations[:,:3];b=hull.equations[:,3]
            result=linprog(np.zeros(3),A_ub=A@tri.T,b_ub=-b,A_eq=np.ones((1,3)),b_eq=[1.],bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            tested+=1
            if result.success:
                bary=result.x;pt=bary@tri;violation=float(np.max(A@pt+b));assert violation<=1e-8 and abs(float(bary.sum())-1)<1e-9 and min(bary)>=-1e-10
                found={'frame':frame,'geom_id':geom,'triangle_id':int(face),'barycentric':[float(x) for x in bary],'local_point_m':[float(x) for x in pt],
                       'max_halfspace_violation_m':violation,'tested_candidates':tested,'candidate_count':len(candidates)};break
        if found is None:raise ValueError('independent LP did not find intersection for frame '+str(frame))
        witnesses.append(found);print(json.dumps(found),flush=True)
    out={'primary_report_sha256':sha(primary/'report.json'),'pair_rows_sha256':sha(primary/'pairs.jsonl'),'script_sha256':sha(Path(__file__)),
         'frames_and_geometries_checked':len(rows),'six_frame_comparisons':comparisons,'triangle_intersection_lp_witnesses':witnesses,
         'limitations':'Checks hashes, pair coverage, summary, simple convex-vs-surface consistency and finds feasible LP witnesses for four frames. It does not independently certify every lower bound or pruning decision. Mathematical surface/convex intersection at sampled poses does not establish physical contact, force, grasp, correct motion, ground truth or training readiness.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out))
if __name__=='__main__':main()

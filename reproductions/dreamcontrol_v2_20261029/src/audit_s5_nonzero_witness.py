"""Find explicit independent witnesses for nonzero S5 index/cup separation."""
import hashlib,json,sys
from pathlib import Path
import mujoco,numpy as np,trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree,ConvexHull
from scipy.spatial.transform import Rotation
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'grab_hand_independent_all_20261030'))
from grab_hand_independent_distance import bracket
sys.path.insert(0,str(ROOT/'grab_s3_hand_distance_bound_audit_v2_20261030'))
from grab_s3_hand_distance_bound_audit import bounds
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    prev=ROOT/'grab_s5_cup_surface_bounds_20261030';source=ROOT/'grab_s5_cup_human_source_20261030';run=ROOT/'pyroki_grab_s5_cup_windowed_20261030';conv=ROOT/'grab_s5_hand_independent_20261030'
    doc=json.loads((prev/'report.json').read_text());s=json.loads((source/'report.json').read_text());cdoc=json.loads((conv/'report.json').read_text())
    assert sha(prev/'pairs.jsonl')==doc['pair_rows_sha256'] and sha(prev/'run.py')==doc['script_sha256']
    assert sha(source/'human_source.npz')==doc['source_sha256'] and sha(run/'windows.jsonl')==doc['windows_sha256'] and sha(conv/'report.json')==doc['convex_report_sha256']
    assert sha(ROOT/'grab_hand_independent_all_20261030/grab_hand_independent_distance.py')==cdoc['bracket_code_sha256']
    pairs={(r['frame'],r['geom_id']):r for r in map(json.loads,(prev/'pairs.jsonl').read_text().splitlines())};selected={};priority=[1,2,3,4,3,2,1];names=None
    for row in map(json.loads,(run/'windows.jsonl').read_text().splitlines()):
        file=run/row['artifact'];assert sha(file)==row['artifact_sha256']
        with np.load(file,allow_pickle=False) as d:
            if names is None:names=d['joint_names'].tolist()
            for j,f in enumerate(d['frames']):
                f=int(f)
                if f not in selected or priority[j]>selected[f][0]:selected[f]=(priority[j],d['root_wxyz_xyz'][j].copy(),d['joint_pos'][j].copy())
    with np.load(source/'human_source.npz',allow_pickle=False) as d:
        idx=np.searchsorted(d['input_frame_indices'],[367,372]);assert d['input_frame_indices'][idx].tolist()==[367,372]
        orient=d['object_orient_rotvec'][idx];trans=d['object_transl'][idx]
    meshfile=ROOT/'grab_mesh_stage_20261030'/s['object_mesh'];assert sha(meshfile)==doc['mesh_sha256'];mesh=trimesh.load(str(meshfile),process=False)
    vertices=np.asarray(mesh.vertices,float);faces=np.asarray(mesh.faces,int)
    scene=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description/scene_full.xml');assert sha(scene)==doc['scene_sha256']
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    def mid(kind,name):
        a=mujoco.mj_name2id(model,kind,name)
        if a<0:raise ValueError('missing '+name)
        return a
    addr=int(model.jnt_qposadr[mid(mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint')]);joints=[mid(mujoco.mjtObj.mjOBJ_JOINT,n) for n in names];geom=99;witnesses=[]
    for k,frame in enumerate((367,372)):
        _,root,q=selected[frame];mujoco.mj_resetData(model,data);data.qpos[addr:addr+3]=root[4:];data.qpos[addr+3:addr+7]=root[:4]
        for j,val in zip(joints,q):data.qpos[model.jnt_qposadr[j]]=val
        mujoco.mj_forward(model,data);assert mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[geom]))=='right_hand_index_0_link'
        rot=data.geom_xmat[geom].reshape(3,3);p=data.geom_xpos[geom];mi=int(model.geom_dataid[geom]);ad=int(model.mesh_vertadr[mi]);n=int(model.mesh_vertnum[mi]);local=np.asarray(model.mesh_vert[ad:ad+n],float);hand=p+local@rot.T
        world=Rotation.from_rotvec(orient[k]).inv().apply(vertices)+trans[k];tri=world[faces];gl,gh=bounds(model,data,geom);box=np.linalg.norm(np.maximum(0.,np.maximum(tri.min(axis=1)-gh,gl-tri.max(axis=1))),axis=1)
        target=pairs[(frame,geom)]['upper_m'];order=np.argsort(box,kind='stable');chosen=None;tested=0
        for t in order:
            if box[t]>target+3e-6:break
            b=bracket(tri[t],model,data,geom);tested+=1
            if b['upper_m']<=target+2e-7:chosen=int(t);break
        if chosen is None:raise ValueError('could not recover near-minimum triangle')
        # Independent constrained Euclidean minimization over triangle barycentric coordinates
        # and hand convex-hull halfspaces. A feasible pair supplies an explicit upper bound.
        H=ConvexHull(local).equations;triangle=(tri[chosen]-p)@rot;A=H[:,:3];c=H[:,3]
        def diff(z):return z[:3]@triangle-z[3:]
        def fun(z):return float(np.dot(diff(z),diff(z)))
        def jac(z):
            v=2*diff(z);return np.r_[triangle@v,-v]
        x0=np.r_[np.full(3,1/3),local[cKDTree(local).query(triangle.mean(axis=0))[1]]]
        res=minimize(fun,x0,jac=jac,method='SLSQP',bounds=[(0.,1.)]*3+[(None,None)]*3,
            constraints=[{'type':'eq','fun':lambda z:float(z[:3].sum()-1),'jac':lambda z:np.array([1.,1.,1.,0.,0.,0.])},
                         {'type':'ineq','fun':lambda z:-(A@z[3:]+c),'jac':lambda z:np.column_stack((np.zeros((len(A),3)),-A))}],
            options={'maxiter':600,'ftol':1e-15})
        z=res.x;error=float(np.max(A@z[3:]+c));assert res.success and error<=1e-8 and abs(float(sum(z[:3]))-1)<1e-9 and min(z[:3])>=-1e-10,(frame,res.message,error)
        a=p+(z[:3]@triangle)@rot.T;b=p+z[3:]@rot.T;upper=float(np.linalg.norm(a-b));direction=(a-b)/upper
        # Surface is a subset of the convex hull of all cup vertices, so this separating-axis
        # inequality is a global lower bound on every original surface triangle.
        lower=max(0.,float(np.min(world@direction)-np.max(hand@direction)))
        assert lower<=upper+1e-9 and upper<=target+1e-5,(frame,lower,upper,target)
        w={'frame':frame,'geom_id':geom,'triangle_id':chosen,'triangle_barycentric':[float(x) for x in z[:3]],'hand_local_point_m':[float(x) for x in z[3:]],
           'distance_lower_separating_axis_m':lower,'distance_upper_feasible_pair_m':upper,'claimed_pair_lower_m':pairs[(frame,geom)]['lower_m'],'claimed_pair_upper_m':target,
           'hand_hull_max_violation_m':error,'candidate_triangles_tested':tested}
        witnesses.append(w);print(json.dumps(w),flush=True)
    out={'source_sha256':doc['source_sha256'],'mesh_sha256':doc['mesh_sha256'],'primary_report_sha256':sha(prev/'report.json'),'pairs_sha256':sha(prev/'pairs.jsonl'),'script_sha256':sha(Path(__file__)),
         'witnesses':witnesses,'limitations':'Only right index collider in two frames. Candidate triangle was found with the original support solver, then the feasible nearest pair and separating-axis inequality were independently evaluated. Does not certify all 11 colliders or all original pair bounds, real contact/forces, dynamics or training readiness.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out))
if __name__=='__main__':main()

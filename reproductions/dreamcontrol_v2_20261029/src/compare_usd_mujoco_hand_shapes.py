"""Static local-shape inventory for USD and MuJoCo hand collision assets."""
import hashlib
import json
import sys
from pathlib import Path
import mujoco
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

BASE=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
USD=BASE/'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof_with_hands_min_collisions_flat_white.usd'
SCENE=BASE/'Sim2Real/resources/robots/g1_description/scene_full.xml'
INCLUDE=SCENE.parent/'g1_27dof_with_hand_rev_1_0.xml'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def box(h):return np.array([[x,y,z] for x in (-h[0],h[0]) for y in (-h[1],h[1]) for z in (-h[2],h[2])],dtype=float)
def summary(vertices):return {'count':len(vertices),'min_m':vertices.min(axis=0).tolist(),'max_m':vertices.max(axis=0).tolist(),'array_sha256':hashlib.sha256(np.ascontiguousarray(vertices).tobytes()).hexdigest()}
stage=Usd.Stage.Open(str(USD));assert UsdGeom.GetStageMetersPerUnit(stage)==1
model=mujoco.MjModel.from_xml_path(str(SCENE));cache=UsdGeom.XformCache();a={};b={}
for p in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
    if not p.HasAPI(UsdPhysics.CollisionAPI):continue
    path=str(p.GetPath());segments=path.split('/');names=[s for s in segments if s.startswith(('right_wrist_','right_hand_')) and s.endswith('_link')]
    if not names:continue
    body=names[0];parent=stage.GetPrimAtPath('/g1_29dof_with_hand_rev_1_0/'+body)
    if p.IsA(UsdGeom.Mesh): vertices=np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(),dtype=float)
    elif p.IsA(UsdGeom.Cube): vertices=box(np.full(3,float(UsdGeom.Cube(p).GetSizeAttr().Get())/2))
    else:raise ValueError('unexpected USD geometry '+path)
    T=cache.GetLocalToWorldTransform(p)*cache.GetLocalToWorldTransform(parent).GetInverse();T=np.asarray(T,dtype=float)
    vertices=vertices@T[:3,:3]+T[3,:3];api=UsdPhysics.MeshCollisionAPI(p)
    approx=str(api.GetApproximationAttr().Get()) if p.HasAPI(UsdPhysics.MeshCollisionAPI) else None
    a.setdefault(body,[]).append({'path':path,'type':p.GetTypeName(),'collision_enabled':bool(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()),'mesh_approximation':approx,'vertices':vertices})
for i in range(model.ngeom):
    body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[i])) or ''
    if not body.startswith(('right_wrist_','right_hand_')) or not (model.geom_contype[i] or model.geom_conaffinity[i]):continue
    kind=int(model.geom_type[i]);asset=int(model.geom_dataid[i])
    if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
        first=int(model.mesh_vertadr[asset]);length=int(model.mesh_vertnum[asset]);vertices=np.asarray(model.mesh_vert[first:first+length],dtype=float)
    elif kind==int(mujoco.mjtGeom.mjGEOM_BOX):vertices=box(model.geom_size[i])
    else:raise ValueError('unexpected MJ geometry')
    q=model.geom_quat[i];R=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
    vertices=vertices@R.T+model.geom_pos[i]
    b.setdefault(body,[]).append({'geom_id':i,'type':kind,'contype':int(model.geom_contype[i]),'conaffinity':int(model.geom_conaffinity[i]),'vertices':vertices})
comparison=[]
for name in sorted(set(a)&set(b)):
    U=a[name];M=b[name]
    cost=np.array([[np.linalg.norm(u['vertices'].min(axis=0)-m['vertices'].min(axis=0))+np.linalg.norm(u['vertices'].max(axis=0)-m['vertices'].max(axis=0)) for m in M] for u in U])
    for ui,mi in zip(*linear_sum_assignment(cost)):
        u,m=U[ui],M[mi];x,y=u['vertices'],m['vertices'];diff=np.r_[x.min(axis=0)-y.min(axis=0),x.max(axis=0)-y.max(axis=0)]
        u_to_m=cKDTree(y).query(x,workers=1)[0];m_to_u=cKDTree(x).query(y,workers=1)[0]
        comparison.append({'body':name,'usd':{**{k:v for k,v in u.items() if k!='vertices'},'bounds':summary(x)},'mujoco':{**{k:v for k,v in m.items() if k!='vertices'},'bounds':summary(y)},'aabb_max_abs_difference_m':float(np.max(np.abs(diff))),'usd_to_mj_vertex_median_m':float(np.median(u_to_m)),'mj_to_usd_vertex_median_m':float(np.median(m_to_u)),'usd_to_mj_vertex_max_m':float(u_to_m.max()),'mj_to_usd_vertex_max_m':float(m_to_u.max())})
def public(groups):return {name:[{**{k:v for k,v in v.items() if k!='vertices'},'bounds':summary(v['vertices'])} for v in values] for name,values in groups.items()}
out={'usd_sha256':sha(USD),'scene_sha256':sha(SCENE),'scene_include_sha256':sha(INCLUDE),'script_sha256':sha(Path(__file__)),'usd_shapes':public(a),'mujoco_shapes':public(b),'same_body_pairs':comparison,'unmatched_body_names_usd':sorted(set(a)-set(b)),'unmatched_body_names_mujoco':sorted(set(b)-set(a)),'limitations':'Authored USD and compiled MuJoCo collision shapes in their respective local rigid-body frames; static vertex AABBs, point-to-point distances and collision flags are not runtime collision surfaces, filtering, hand FK, forces or acceptance. MuJoCo input meshes are represented by compiled geometry hashes; not a complete asset provenance inventory. Point-set differences can reflect tessellation.'}
with Path(sys.argv[1]).open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)

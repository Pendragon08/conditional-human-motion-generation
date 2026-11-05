"""Audit local collision bounds by independent USD/MuJoCo transform paths."""
import hashlib
import json
import sys
from pathlib import Path
import mujoco
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
BASE=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
USD=BASE/'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof_with_hands_min_collisions_flat_white.usd'
SCENE=BASE/'Sim2Real/resources/robots/g1_description/scene_full.xml';INCLUDE=SCENE.parent/'g1_27dof_with_hand_rev_1_0.xml'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def box(h):return np.array([[x,y,z] for x in (-h[0],h[0]) for y in (-h[1],h[1]) for z in (-h[2],h[2])],float)
rpath=Path(sys.argv[1]);r=json.loads(rpath.read_text());assert sha(USD)==r['usd_sha256'] and sha(SCENE)==r['scene_sha256'] and sha(INCLUDE)==r['scene_include_sha256']
s=Usd.Stage.Open(str(USD));model=mujoco.MjModel.from_xml_path(str(SCENE));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
usd_paths=set()
for p in Usd.PrimRange.Stage(s,Usd.TraverseInstanceProxies()):
    if p.HasAPI(UsdPhysics.CollisionAPI) and any(x.startswith(('right_wrist_','right_hand_')) and x.endswith('_link') for x in str(p.GetPath()).split('/')):
        usd_paths.add(str(p.GetPath()))
assert usd_paths=={item['path'] for vals in r['usd_shapes'].values() for item in vals}
ids=set()
for i in range(model.ngeom):
    body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[i])) or ''
    if body.startswith(('right_wrist_','right_hand_')) and (model.geom_contype[i] or model.geom_conaffinity[i]):ids.add(i)
assert ids=={item['geom_id'] for vals in r['mujoco_shapes'].values() for item in vals}
maxerr=0.
for pair in r['same_body_pairs']:
    body=pair['body'];u=pair['usd'];m=pair['mujoco'];p=s.GetPrimAtPath(u['path']);assert p.HasAPI(UsdPhysics.CollisionAPI)
    if p.IsA(UsdGeom.Mesh):v=np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(),dtype=float)
    else:assert p.IsA(UsdGeom.Cube);v=box(np.full(3,float(UsdGeom.Cube(p).GetSizeAttr().Get())/2))
    parent='/g1_29dof_with_hand_rev_1_0/'+body;T=Gf.Matrix4d(1)
    while str(p.GetPath())!=parent:
        x=UsdGeom.Xformable(p);assert not x.GetResetXformStack();T=T*x.GetLocalTransformation();p=p.GetParent();assert p
    T=np.asarray(T,dtype=float);v=v@T[:3,:3]+T[3,:3]
    uerr=max(np.max(np.abs(v.min(axis=0)-u['bounds']['min_m'])),np.max(np.abs(v.max(axis=0)-u['bounds']['max_m'])))
    i=m['geom_id'];bid=int(model.geom_bodyid[i]);assert mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,bid)==body
    if int(model.geom_type[i])==int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh=int(model.geom_dataid[i]);a=int(model.mesh_vertadr[mesh]);n=int(model.mesh_vertnum[mesh]);v=np.asarray(model.mesh_vert[a:a+n],float)
    else:assert int(model.geom_type[i])==int(mujoco.mjtGeom.mjGEOM_BOX);v=box(model.geom_size[i])
    world=v@data.geom_xmat[i].reshape(3,3).T+data.geom_xpos[i];local=(world-data.xpos[bid])@data.xmat[bid].reshape(3,3)
    merr=max(np.max(np.abs(local.min(axis=0)-m['bounds']['min_m'])),np.max(np.abs(local.max(axis=0)-m['bounds']['max_m'])))
    maxerr=max(maxerr,float(uerr),float(merr));assert uerr<1e-8 and merr<1e-8
assert len(r['same_body_pairs'])==9 and r['unmatched_body_names_mujoco']==['right_wrist_pitch_link','right_wrist_roll_link']
out={'status':'passed','report_sha256':sha(rpath),'usd_sha256':sha(USD),'scene_sha256':sha(SCENE),'scene_include_sha256':sha(INCLUDE),'shapes_checked_each_side':9,'max_aabb_component_error_m':maxerr,'script_sha256':sha(Path(__file__)),'limitations':'Verifies matched AABBs through alternate USD transform chain and MuJoCo forward-derived geom/body transforms only; same model parser, not complete geometry or actual PhysX contact verification.'}
with Path(sys.argv[2]).open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)

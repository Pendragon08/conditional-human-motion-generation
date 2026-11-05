"""Investigate impossible zero distances; bounded CPU diagnostic, not acceptance."""
import json, sys, zipfile
from pathlib import Path
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0, str(ROOT/'grab_s3_scene_timeline_v2_20261030'))
from grab_s3_scene_timeline import sha, load_table, mesh_for
sys.path.insert(0, str(ROOT/'grab_s3_human_source_20261030'))
from grab_restricted_reader import read_grab_field

def make_model(hulls):
    spec=mujoco.MjSpec()
    for n,h in hulls.items():
        mesh=spec.add_mesh(name=n+'_hull')
        mesh.uservert=np.asarray(h.vertices,dtype=np.float64).reshape(-1).tolist()
        mesh.userface=np.asarray(h.faces,dtype=np.int32).reshape(-1).tolist()
        b=spec.worldbody.add_body(name=n,mocap=True)
        b.add_geom(name=n,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=n+'_hull',contype=1,conaffinity=1)
    return spec.compile()

def main():
    meta=json.loads((ROOT/'grab_s3_human_source_20261030/report.json').read_text())
    archive=ROOT/'data/GRAB/grab__s3.zip'
    if sha(archive)!=meta['archive_sha256']:raise ValueError('archive changed')
    with zipfile.ZipFile(archive) as z:obj=read_grab_field(z,meta['source_member'],'object')
    table=load_table(archive,meta['source_member'])
    ap,ao,ah=mesh_for(obj['object_mesh']);tp,to,th=mesh_for(table['table_mesh'])
    # Deterministic directions select vertices from the full convex hull. The
    # resulting smaller hull is an INSCRIBED approximation; it is not the same
    # object. Its extreme support error bounds are also recorded.
    count=512; i=np.arange(count);z=1-2*(i+.5)/count
    theta=i*np.pi*(3-np.sqrt(5)); dirs=np.column_stack((np.sqrt(1-z*z)*np.cos(theta),np.sqrt(1-z*z)*np.sin(theta),z))
    vertices=np.asarray(ah.vertices); chosen=[]
    for d in dirs:chosen.append(int(np.argmax(vertices@d)))
    small=trimesh.Trimesh(vertices=vertices[np.unique(chosen)],process=False).convex_hull
    error=[]
    for d in dirs:error.append(float(np.max(vertices@d)-np.max(np.asarray(small.vertices)@d)))
    frames=[277,278,279,294,295,296,355,356,357,359,360,361,364,365,366]
    solutions=[]
    for label,apple in (('full',ah),('inscribed_support_512',small)):
        m=make_model({'apple':apple,'table':th});data=mujoco.MjData(m)
        gids={n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('apple','table')}
        mids={n:int(m.body_mocapid[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)]) for n in gids}
        for frame in frames:
            for n,entry in (('apple',obj),('table',table)):
                orient=np.asarray(entry['params']['global_orient'])[frame]
                trans=np.asarray(entry['params']['transl'])[frame]
                q=Rotation.from_rotvec(orient).inv().as_quat()
                data.mocap_pos[mids[n]]=trans;data.mocap_quat[mids[n]]=q[[3,0,1,2]]
            for mode in ('native','libccd'):
                if mode=='libccd':m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
                else:m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
                mujoco.mj_forward(m,data)
                dist=float(mujoco.mj_geomDistance(m,data,gids['apple'],gids['table'],2.,None))
                boxes={}
                for n,entry,h in (('apple',obj,apple),('table',table,th)):
                    world=Rotation.from_rotvec(np.asarray(entry['params']['global_orient'])[frame]).inv().apply(h.vertices)+np.asarray(entry['params']['transl'])[frame]
                    boxes[n]=(world.min(axis=0),world.max(axis=0))
                lower=float(np.linalg.norm(np.maximum(0.,np.maximum(boxes['apple'][0]-boxes['table'][1],boxes['table'][0]-boxes['apple'][1]))))
                solutions.append({'frame':frame,'mesh':label,'solver':mode,'distance_m':dist,'aabb_lower_bound_m':lower,
                                  'aabb_violation':bool(dist+1e-6<lower)})
    report={'input_sha256':{'archive':sha(archive),'apple':sha(ap),'table':sha(tp),'script':sha(Path(__file__))},
            'version':mujoco.__version__,'full_hull_vertices':len(ah.vertices),'small_hull_vertices':len(small.vertices),
            'directional_support_error_m_max_on_sampling_directions':max(error),
            'results':solutions,'warning':'15 selected frames; independent AABB bounds, no physical forces. Smaller apple hull is an inscribed approximation; tested support directions are not a global error bound. Backend selection is diagnostic and does not establish general solver correctness.'}
    out=Path(__file__).with_name('report.json')
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'output':str(out),'summary':{str((mesh,mode)):sum(r['aabb_violation'] for r in solutions if r['mesh']==mesh and r['solver']==mode) for mesh in ('full','inscribed_support_512') for mode in ('native','libccd')}}),flush=True)

if __name__=='__main__':main()

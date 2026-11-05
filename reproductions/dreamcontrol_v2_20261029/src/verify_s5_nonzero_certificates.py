"""Reconstruct feasible points and separating planes without solving optimization."""
import hashlib,json
from pathlib import Path
import mujoco,numpy as np,trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');PREV=ROOT/'grab_s5_nonzero_witness_20261030';SRC=ROOT/'grab_s5_cup_human_source_20261030';RUN=ROOT/'pyroki_grab_s5_cup_windowed_20261030';OUT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    r=json.loads((PREV/'report.json').read_text());sr=json.loads((SRC/'report.json').read_text());assert sha(PREV/'audit.py')==r['script_sha256'] and sha(SRC/'human_source.npz')==r['source_sha256']
    meshpath=ROOT/'grab_mesh_stage_20261030'/sr['object_mesh'];assert sha(meshpath)==r['mesh_sha256'];mesh=trimesh.load(str(meshpath),process=False)
    vertices=np.asarray(mesh.vertices,float);faces=np.asarray(mesh.faces,int);scene=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description/scene_full.xml')
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model);priorities=[1,2,3,4,3,2,1];best={};names=None
    for row in map(json.loads,(RUN/'windows.jsonl').read_text().splitlines()):
        path=RUN/row['artifact'];assert sha(path)==row['artifact_sha256']
        with np.load(path,allow_pickle=False) as d:
            if names is None:names=d['joint_names'].tolist()
            for j,f in enumerate(d['frames']):
                f=int(f)
                if f not in best or priorities[j]>best[f][0]:best[f]=(priorities[j],d['root_wxyz_xyz'][j].copy(),d['joint_pos'][j].copy())
    with np.load(SRC/'human_source.npz',allow_pickle=False) as d:
        inds=np.searchsorted(d['input_frame_indices'],[367,372]);rotvec=d['object_orient_rotvec'][inds].copy();trans=d['object_transl'][inds].copy()
    def mid(kind,name):
        val=mujoco.mj_name2id(model,kind,name)
        if val<0:raise ValueError(name)
        return val
    addr=int(model.jnt_qposadr[mid(mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint')]);jointids=[mid(mujoco.mjtObj.mjOBJ_JOINT,n) for n in names];checks=[]
    for k,w in enumerate(r['witnesses']):
        frame=w['frame'];_,root,q=best[frame];mujoco.mj_resetData(model,data);data.qpos[addr:addr+3]=root[4:];data.qpos[addr+3:addr+7]=root[:4]
        for j,val in zip(jointids,q):data.qpos[model.jnt_qposadr[j]]=val
        mujoco.mj_forward(model,data);g=w['geom_id'];p=data.geom_xpos[g];R=data.geom_xmat[g].reshape(3,3);mi=int(model.geom_dataid[g]);st=int(model.mesh_vertadr[mi]);count=int(model.mesh_vertnum[mi]);handlocal=np.asarray(model.mesh_vert[st:st+count],float)
        hand=p+handlocal@R.T;world=vertices@Rotation.from_rotvec(rotvec[k]).as_matrix()+trans[k];bary=np.array(w['triangle_barycentric']);hp=np.array(w['hand_local_point_m'])
        assert abs(bary.sum()-1)<1e-9 and np.min(bary)>=-1e-10
        violation=float(np.max(ConvexHull(handlocal).equations[:,:3]@hp+ConvexHull(handlocal).equations[:,3]));assert violation<1e-8
        a=bary@world[faces[w['triangle_id']]];b=p+hp@R.T;delta=a-b;upper=float(np.linalg.norm(delta));direction=delta/upper
        lower=max(0.,float(np.min(world@direction)-np.max(hand@direction)))
        assert abs(lower-w['distance_lower_separating_axis_m'])<1e-10 and abs(upper-w['distance_upper_feasible_pair_m'])<1e-10 and lower<=upper+1e-10
        checks.append({'frame':frame,'lower_m':lower,'upper_m':upper,'max_hand_halfspace_violation_m':violation})
    result={'witness_report_sha256':sha(PREV/'report.json'),'script_sha256':sha(Path(__file__)),'checks':checks,'limitations':'Independent certificate reconstruction for one index collider at two frames; full hand lower bounds, real contact and dynamics not established.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))
if __name__=='__main__':main()

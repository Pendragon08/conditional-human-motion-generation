"""Audit S5 bounds with independently reconstructed selection and AABB lower bounds."""
import hashlib,json,sys
from pathlib import Path
import mujoco,numpy as np,trimesh
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
    prev=ROOT/'grab_s5_hand_independent_20261030';run=ROOT/'pyroki_grab_s5_cup_windowed_20261030';source=ROOT/'grab_s5_cup_human_source_20261030'
    doc=json.loads((prev/'report.json').read_text());sr=json.loads((source/'report.json').read_text());r=json.loads((run/'report.json').read_text())
    assert sha(prev/'frames.jsonl')==doc['frame_results_sha256'] and sha(prev/'run.py')==doc['script_sha256']
    assert sha(source/'human_source.npz')==doc['source_sha256'] and sha(run/'windows.jsonl')==doc['rows_source_sha256'] and sha(run/'report.json')==doc['run_report_sha256']
    result=[json.loads(x) for x in (prev/'frames.jsonl').read_text().splitlines()];rows=[json.loads(x) for x in (run/'windows.jsonl').read_text().splitlines()]
    selected={};weights=[1,2,3,4,3,2,1];names=None
    for row in rows:
        path=run/row['artifact'];assert sha(path)==row['artifact_sha256']
        with np.load(path,allow_pickle=False) as d:
            if names is None:names=d['joint_names'].tolist()
            assert d['joint_names'].tolist()==names
            for i,frame in enumerate(d['frames']):
                frame=int(frame)
                if frame not in selected or weights[i]>selected[frame][0]:selected[frame]=(weights[i],d['root_wxyz_xyz'][i].copy(),d['joint_pos'][i].copy())
    assert sorted(selected)==list(range(309,427)) and [x['frame'] for x in result]==list(range(309,427))
    with np.load(source/'human_source.npz',allow_pickle=False) as d:
        index=np.searchsorted(d['input_frame_indices'],np.arange(309,427));assert np.array_equal(d['input_frame_indices'][index],np.arange(309,427))
        orient=d['object_orient_rotvec'][index].copy();trans=d['object_transl'][index].copy();labels=d['right_hand_labels'][index].copy()
    meshpath=ROOT/'grab_mesh_stage_20261030'/sr['object_mesh'];assert sha(meshpath)==doc['mesh_sha256']
    vertices=np.asarray(trimesh.load(str(meshpath),process=False).convex_hull.vertices,dtype=float)
    scene=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description/scene_full.xml');assert sha(scene)==doc['scene_sha256']
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    def get(kind,n):
        i=mujoco.mj_name2id(model,kind,n)
        if i<0:raise ValueError('missing '+n)
        return i
    raddr=int(model.jnt_qposadr[get(mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint')]);joints=[get(mujoco.mjtObj.mjOBJ_JOINT,n) for n in names];hand=[]
    for g in range(model.ngeom):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or ''
        if name.startswith(('right_hand_','right_wrist_')) and (model.geom_contype[g] or model.geom_conaffinity[g]):hand.append(g)
    assert len(hand)==11
    violations=[];bounds_min=[]
    for i,row in enumerate(result):
        _,root,q=selected[row['frame']];mujoco.mj_resetData(model,data);data.qpos[raddr:raddr+3]=root[4:];data.qpos[raddr+3:raddr+7]=root[:4]
        for j,val in zip(joints,q):data.qpos[model.jnt_qposadr[j]]=val
        mujoco.mj_forward(model,data);v=Rotation.from_rotvec(orient[i]).inv().apply(vertices)+trans[i];low=v.min(0);high=v.max(0)
        ds=[]
        for g in hand:
            gl,gh=bounds(model,data,g);ds.append(float(np.linalg.norm(np.maximum(0.,np.maximum(low-gh,gl-high)))))
        boxmin=min(ds);bounds_min.append(boxmin)
        if boxmin>row['upper_m']+1e-7:violations.append(row['frame'])
        assert row['right_hand_label']==bool(labels[i]) and 0<=row['lower_m']<=row['upper_m']+1e-9
    lowers=np.array([x['lower_m'] for x in result]);uppers=np.array([x['upper_m'] for x in result]);assert not violations
    assert abs(float(np.median(lowers))-doc['median_lower_m'])<1e-12 and abs(float(np.median(uppers))-doc['median_upper_m'])<1e-12
    assert sum(x['nonbounded_pairs'] for x in result)==doc['nonbounded_pairs'] and sum(lowers==0)==doc['zero_lower_frames']
    out={'primary_report_sha256':sha(prev/'report.json'),'frames_sha256':sha(prev/'frames.jsonl'),'script_sha256':sha(Path(__file__)),
         'aabb_code_sha256':sha(ROOT/'grab_s3_hand_distance_bound_audit_v2_20261030/grab_s3_hand_distance_bound_audit.py'),
         'frames_checked':len(result),'aabb_lower_greater_than_upper_frames':violations,'largest_aabb_lower_m':max(bounds_min),
         'zero_lower_frames':[row['frame'] for row in result if row['lower_m']==0],
         'limitations':'AABB is a conservative lower bound, so absence of contradictions does not establish accuracy or contact. Independent frame selection and geometry bounds; FK and cup convex hull are shared assumptions. Cup concavity, ground, force and physical grasp are not checked.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out))
if __name__=='__main__':main()

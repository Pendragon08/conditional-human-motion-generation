"""Experimental S5 synchronized hand/cup convex-distance bounds; no physics verdict."""
import hashlib,json,sys
from pathlib import Path
import mujoco,numpy as np,trimesh
from scipy.spatial.transform import Rotation
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'grab_hand_independent_all_20261030'))
from grab_hand_independent_distance import bracket
sys.path.insert(0,str(ROOT/'grab_s3_object_wrist_audit_v2_20261030'))
from grab_s3_object_wrist_audit_v2 import chosen,load
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    src=ROOT/'grab_s5_cup_human_source_20261030';run=ROOT/'pyroki_grab_s5_cup_windowed_20261030'
    sr=json.loads((src/'report.json').read_text());rr=json.loads((run/'report.json').read_text());audit=json.loads((run/'audit_report.json').read_text())
    if sha(src/'human_source.npz')!=sr['artifact_sha256'] or rr['source_sha256']!=sr['artifact_sha256'] or sha(run/'windows.jsonl')!=rr['rows_sha256'] or sha(run/'report.json')!=audit['report_sha256']:raise ValueError('source or window provenance')
    rows=[json.loads(x) for x in (run/'windows.jsonl').read_text().splitlines()];assert len(rows)==29
    windows=[load(run/x['artifact'],x['artifact_sha256']) for x in rows];frames,roots,q=chosen(windows)
    if not np.array_equal(frames,np.arange(309,427)):raise ValueError('frame coverage')
    names=windows[0]['joint_names'].tolist()
    with np.load(src/'human_source.npz',allow_pickle=False) as f:
        ind=np.searchsorted(f['input_frame_indices'],frames)
        if not np.array_equal(f['input_frame_indices'][ind],frames) or float(f['input_fps'])!=120:raise ValueError('timeline mismatch')
        orient=f['object_orient_rotvec'][ind];trans=f['object_transl'][ind];labels=f['right_hand_labels'][ind]
    meshfile=ROOT/'grab_mesh_stage_20261030'/sr['object_mesh']
    if sha(meshfile)!=sr['object_mesh_sha256']:raise ValueError('cup mesh hash mismatch')
    mesh=trimesh.load(str(meshfile),process=False);hull=mesh.convex_hull;vertices=np.asarray(hull.vertices,dtype=float)
    scene=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description/scene_full.xml')
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    def mid(kind,name):
        x=mujoco.mj_name2id(model,kind,name)
        if x<0:raise ValueError('missing '+name)
        return x
    rid=mid(mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint');raddr=int(model.jnt_qposadr[rid]);joints=[mid(mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
    hand=[]
    for g in range(model.ngeom):
        bn=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or ''
        if bn.startswith(('right_hand_','right_wrist_')) and (model.geom_contype[g]!=0 or model.geom_conaffinity[g]!=0):hand.append(g)
    if len(hand)!=11:raise ValueError('unexpected hand geometry count')
    simple=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="s" type="sphere" size=".1" pos=".7 0 0"/></worldbody></mujoco>');sd=mujoco.MjData(simple);mujoco.mj_forward(simple,sd)
    box=np.array([[x,y,z] for x in (-.1,.1) for y in (-.1,.1) for z in (-.1,.1)]);test=bracket(box,simple,sd,0)
    if not (test['lower_m']-1e-8<=.5<=test['upper_m']+1e-8 and test['gap_m']<=2e-5):raise ValueError('synthetic bound failed')
    results=[];transform_max=0.
    with (OUT/'frames.jsonl').open('x',encoding='utf8') as f:
        for i,frame in enumerate(frames):
            rotation=Rotation.from_rotvec(orient[i]);world=vertices@rotation.as_matrix()+trans[i]
            alt=rotation.inv().apply(vertices)+trans[i];transform_max=max(transform_max,float(np.max(np.abs(world-alt))))
            mujoco.mj_resetData(model,data);data.qpos[raddr:raddr+3]=roots[i,4:];data.qpos[raddr+3:raddr+7]=roots[i,:4]
            for j,value in zip(joints,q[i]):data.qpos[model.jnt_qposadr[j]]=value
            data.qvel[:]=0;mujoco.mj_forward(model,data)
            bounds=[bracket(world,model,data,g) for g in hand]
            rec={'frame':int(frame),'right_hand_label':bool(labels[i]),'lower_m':min(x['lower_m'] for x in bounds),'upper_m':min(x['upper_m'] for x in bounds),
                 'nonbounded_pairs':sum(x['status']!='bounded' for x in bounds),'max_pair_gap_m':max(x['gap_m'] for x in bounds)}
            if rec['lower_m']>rec['upper_m']+1e-9:raise ValueError('invalid bound')
            f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();results.append(rec)
            print(json.dumps({'completed_frame':int(frame)}),flush=True)
    lo=np.array([x['lower_m'] for x in results]);hi=np.array([x['upper_m'] for x in results])
    out={'source_report_sha256':sha(src/'report.json'),'source_sha256':sha(src/'human_source.npz'),'run_report_sha256':sha(run/'report.json'),'rows_source_sha256':sha(run/'windows.jsonl'),
         'mesh_sha256':sha(meshfile),'scene_sha256':sha(scene),'bracket_code_sha256':sha(ROOT/'grab_hand_independent_all_20261030/grab_hand_independent_distance.py'),'script_sha256':sha(Path(__file__)),'frame_results_sha256':sha(OUT/'frames.jsonl'),
         'frames':len(results),'geom_pairs':len(results)*len(hand),'right_labeled_frames':sum(x['right_hand_label'] for x in results),
         'minimum_lower_m':float(lo.min()),'minimum_upper_m':float(hi.min()),'median_lower_m':float(np.median(lo)),'median_upper_m':float(np.median(hi)),
         'zero_lower_frames':int((lo==0).sum()),'nonbounded_pairs':sum(x['nonbounded_pairs'] for x in results),'max_pair_gap_m':max(x['max_pair_gap_m'] for x in results),
         'cup_source_vertices':len(mesh.vertices),'cup_convex_hull_vertices':len(vertices),'cup_mesh_is_convex':bool(mesh.is_convex),'transform_row_vs_inverse_max_component_m':transform_max,'synthetic_box_sphere_bound':test,
         'warning':'One S5 clip, center-priority independent unanchored windows, fixed open hand, convex hull fills cup concavity; unsigned bounds on convex sets with MuJoCo FK, not MuJoCo distance query. Ground, force, contact, physical motion and true FPS unverified; not author v2, grasp success or training acceptance.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()

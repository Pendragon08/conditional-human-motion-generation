"""Observe legacy PyRoki root/offset decomposition on a fixed S4 window."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import yourdfpy

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OUT=Path(__file__).resolve().parent
BASE=ROOT/'pyroki_grab_temporal_20261030'
PROJECT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
sys.path.insert(0,str(OUT));sys.path.insert(1,str(ROOT/'foot_contact_fit_20261030'))
from retarget_helpers._utils import get_humanoid_retarget_indices_g1_27dof,create_conn_tree,solve_retargeting,foot_detect
from g1_fk import URDFFK

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    if any(x.platform!='cpu' for x in jax.devices()):raise ValueError('CPU required')
    man=json.loads((OUT/'manifest.json').read_text());receipt=json.loads((BASE/'report.json').read_text())
    if sha(OUT/'retarget_helpers/_utils.py')!=man['instrumented_helper_sha256'] or sha(BASE/'report.json')!=man['base_report_sha256']:raise ValueError('helper changed')
    source=ROOT/'grab_s4_drink_human_source_20261030/human_source.npz';src=json.loads((source.parent/'report.json').read_text())
    if sha(source)!=src['artifact_sha256']:raise ValueError('source changed')
    baseline=ROOT/'pyroki_grab_s4_drink_windowed_20261030/window_00.npz'
    rows=[json.loads(x) for x in (baseline.parent/'windows.jsonl').read_text().splitlines()]
    if sha(baseline)!=rows[0]['artifact_sha256']:raise ValueError('baseline changed')
    with np.load(source,allow_pickle=False) as data:
        human=data['human_joints'].astype(np.float32);frames=data['input_frame_indices'].copy();fps=float(data['input_fps'])
    if human.shape!=(120,22,3) or fps!=120:raise ValueError('source timeline')
    urdf_path=PROJECT/'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
    if sha(urdf_path)!=receipt['urdf_sha256']:raise ValueError('URDF changed')
    urdf=yourdfpy.URDF.load(str(urdf_path),load_meshes=False,load_collision_meshes=True)
    robot=pk.Robot.from_urdf(urdf);coll=pk.collision.RobotCollision.from_urdf(urdf)
    hi,ri=get_humanoid_retarget_indices_g1_27dof();mask=create_conn_tree(robot,ri)
    plane=pk.collision.Heightmap(pose=jaxlie.SE3.identity(),size=jnp.array([.01,.01,1.]),height_data=jnp.zeros((100,100),dtype=jnp.float32))
    w=human[1:9];l,r=foot_detect(w[None],thres=(.05/fps)**2);target=jnp.asarray(w[1:])
    final,joints,offset,raw=solve_retargeting(robot=robot,robot_coll=coll,target_keypoints=target,
        is_left_foot_contact=jnp.asarray(l[0]>0),is_right_foot_contact=jnp.asarray(r[0]>0),
        left_foot_keypoints=plane.project_points(target[:,10,:]),right_foot_keypoints=plane.project_points(target[:,11,:]),
        smpl_joint_retarget_indices=hi,g1_joint_retarget_indices=ri,smpl_mask=mask,heightmap=plane,
        weights=receipt['weights'],pick_point=jnp.zeros(3))
    q=np.asarray(joints);post=np.asarray(final.wxyz_xyz);pre=np.asarray(raw.wxyz_xyz);delta=np.asarray(offset)
    with np.load(baseline,allow_pickle=False) as data:
        oldq=data['joint_pos'];oldroot=data['root_wxyz_xyz'];oldframes=data['frames']
    if oldframes.tolist()!=frames[2:9].tolist() or q.shape!=(7,27) or pre.shape!=(7,7) or post.shape!=(7,7):raise ValueError('unexpected shape')
    if delta.shape not in ((7,3),(3,)):raise ValueError('unexpected offset shape')
    delta=np.broadcast_to(delta,(7,3))
    if not np.allclose(post[:,4:],pre[:,4:]+delta,atol=1e-6):raise ValueError('translation composition mismatch')
    fk=URDFFK(urdf_path);names=list(robot.joints.actuated_names);links=[robot.links.names[int(i)] for i in np.asarray(ri)]
    prior=fk.forward(pre[:,4:],pre[:,[1,2,3,0]],q,names,links=links)
    after=fk.forward(post[:,4:],post[:,[1,2,3,0]],q,names,links=links)
    target_np=np.asarray(target)[:,np.asarray(hi),:]
    before=float(np.linalg.norm(prior-target_np,axis=-1).mean());after_error=float(np.linalg.norm(after-target_np,axis=-1).mean())
    artifact=OUT/'diagnostic.npz'
    with artifact.open('xb') as f:np.savez_compressed(f,frames=oldframes,offset=delta,raw_root=pre,final_root=post,joints=q)
    report={'source_sha256':sha(source),'baseline_sha256':sha(baseline),'driver_sha256':sha(__file__),'artifact_sha256':sha(artifact),
            'instrumented_helper_sha256':man['instrumented_helper_sha256'],'frames':oldframes.tolist(),
            'offset_xyz_m':delta.tolist(),'max_offset_time_variation_m':float(np.max(np.abs(delta-delta[0]))),
            'max_baseline_root_position_difference_m':float(np.max(np.linalg.norm(post[:,4:]-oldroot[:,4:],axis=-1))),
            'max_baseline_joint_difference_rad':float(np.max(np.abs(q-oldq))),
            'unscaled_world_keypoint_distance_before_offset_mean_m':before,'unscaled_world_keypoint_distance_after_offset_mean_m':after_error,
            'limitations':'One fixed 7-frame S4 window, old v1 approximate solver and assumed ground. Instrumentation only returns hidden variables; any difference from baseline limits inference. Unscaled point distance is not physical truth. No dynamics, real contact or training acceptance.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()

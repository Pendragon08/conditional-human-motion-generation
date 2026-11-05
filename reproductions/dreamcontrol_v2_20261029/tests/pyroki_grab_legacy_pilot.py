"""Short native-rate GRAB -> G1 legacy PyRoki pilot; never training data."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import yourdfpy

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PROJECT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
sys.path.insert(0,str(PROJECT/'TrajGen/sample'))
from retarget_helpers._utils import (get_humanoid_retarget_indices_g1_27dof,
                                    create_conn_tree, solve_retargeting, foot_detect)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    if any(x.platform!='cpu' for x in jax.devices()): raise ValueError('CPU required')
    source=ROOT/'grab_human_robot_pilot_20261030/pilot_arrays.npz'
    receipt=json.loads((source.parent/'report.json').read_text())
    if sha(source)!=receipt['artifact_sha256']: raise ValueError('human source hash mismatch')
    with np.load(source,allow_pickle=False) as data:
        keypoints=data['human_joints'][:8].astype(np.float32)
        frames=data['input_frame_indices'][:8].copy()
        fps=float(data['input_fps'])
    if keypoints.shape!=(8,22,3) or fps!=120. or not np.all(np.diff(frames)==1):
        raise ValueError('expected contiguous 120 FPS human samples')
    urdf_path=PROJECT/'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
    urdf=yourdfpy.URDF.load(str(urdf_path),load_meshes=False,load_collision_meshes=True)
    robot=pk.Robot.from_urdf(urdf)
    collision=pk.collision.RobotCollision.from_urdf(urdf)
    print('robot/collision loaded',flush=True)
    human_idx,robot_idx=get_humanoid_retarget_indices_g1_27dof()
    mask=create_conn_tree(robot,robot_idx)
    # Legacy foot detector outputs counts in [0,2]; an explicit velocity proxy
    # here converts them to booleans. The threshold is not paper-author data.
    left,right=foot_detect(keypoints[None],thres=(.05/fps)**2)
    left=jnp.asarray(left[0]>0); right=jnp.asarray(right[0]>0)
    target=jnp.asarray(keypoints[1:])
    plane=pk.collision.Heightmap(pose=jaxlie.SE3.identity(),size=jnp.array([.01,.01,1.]),
                                 height_data=jnp.zeros((100,100),dtype=jnp.float32))
    foot_l=plane.project_points(target[:,10,:]); foot_r=plane.project_points(target[:,11,:])
    weights={'local_alignment':2.,'global_alignment':1.,'floor_contact':1.,
             'root_smoothness':2.,'foot_skating':2.,'world_collision':1.}
    print('starting CPU solver',flush=True)
    root,joints=solve_retargeting(robot=robot,robot_coll=collision,target_keypoints=target,
       is_left_foot_contact=left,is_right_foot_contact=right,left_foot_keypoints=foot_l,
       right_foot_keypoints=foot_r,smpl_joint_retarget_indices=human_idx,
       g1_joint_retarget_indices=robot_idx,smpl_mask=mask,heightmap=plane,weights=weights,
       pick_point=jnp.zeros(3))
    q=np.asarray(joints); root_pose=np.asarray(root.wxyz_xyz)
    if q.shape!=(7,27) or root_pose.shape!=(7,7) or not np.isfinite(q).all() or not np.isfinite(root_pose).all():
        raise ValueError('invalid solver output')
    artifact=Path(__file__).with_name('raw_pilot.npz')
    with artifact.open('xb') as f:
        np.savez_compressed(f,frames=frames[1:],fps=np.array(fps),human_keypoints=keypoints[1:],
                            robot_root_wxyz_xyz=root_pose,robot_joint_pos=q,
                            robot_joint_names=np.asarray(robot.joints.actuated_names))
    doc={'source_sha256':sha(source),'urdf_sha256':sha(urdf_path),'script_sha256':sha(__file__),
         'helper_sha256':sha(PROJECT/'TrajGen/sample/retarget_helpers/_utils.py'),
         'artifact_sha256':sha(artifact),'source_frames':frames[1:].tolist(),'fps':fps,
         'human_indices':np.asarray(human_idx).tolist(),'robot_indices':np.asarray(robot_idx).tolist(),
         'left_low_speed_intervals':int(np.asarray(left).sum()),'right_low_speed_intervals':int(np.asarray(right).sum()),
         'weights':weights,'devices':[str(d) for d in jax.devices()],
         'warning':'Seven consecutive frames of a GRAB subject; CPU legacy v1 PyRoki with assumed z=0 and low-speed contacts, v1 loss weights and collision geometry, no object, no author-v2 exact config. Not validated physical or training data.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf8') as f:
        json.dump(doc,f,indent=2,allow_nan=False)
    print('complete',flush=True)


if __name__=='__main__': main()

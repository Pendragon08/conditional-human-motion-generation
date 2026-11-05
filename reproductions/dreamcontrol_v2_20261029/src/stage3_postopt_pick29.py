"""Fit generated Pick motions to all 29 joints with scene hand/stance constraints.

Only generated candidates are processed. Capsule self distances are diagnostics:
the neutral-pose audit demonstrates invariant fixed-link overlaps, so they are
not misrepresented as physical penetrations. Native scene replay remains separate.
"""
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation,Slerp
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import pyroki as pk
import yourdfpy

S=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_pick29_20261103')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--limit-weight',type=float,default=1000.)
    parser.add_argument('--initial-state',type=Path)
    args=parser.parse_args()
    assert not args.output_dir.exists()
    import hashlib,sys
    sys.path.insert(0,str(S.parent/'training_disk_guard_20261101'))
    from g1_fk import PAPER_22_LINKS
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    with np.load(args.sample,allow_pickle=False) as f:
        points=f['canonical_points'][0];goal=f['target_points'];metadata=json.loads(f['metadata_utf8'].tobytes())
    n=len(points);fps=20.
    rotation=np.array(metadata['canonical_to_world_rotation']);translation=np.array(metadata['canonical_to_world_translation'])
    target=points@rotation.T+translation;goals=goal@rotation.T+translation
    urdf=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
    assert sha(urdf)=='572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95'
    u=yourdfpy.URDF.load(str(urdf),load_meshes=False,load_collision_meshes=True)
    robot=pk.Robot.from_urdf(u);names=list(robot.joints.actuated_names)
    links=jnp.array([robot.links.names.index(name) for name in PAPER_22_LINKS])
    feet=jnp.array([robot.links.names.index(side+'_ankle_roll_link') for side in ('left','right')])
    wrist=robot.links.names.index('right_wrist_yaw_link')
    q0=np.zeros(29,dtype=np.float32)
    for side in ('left','right'):
        for part,value in [('hip_pitch',-.2),('knee',.42),('ankle_pitch',-.23),('elbow',.87),('shoulder_pitch',.35)]:
            q0[names.index(side+'_'+part+'_joint')]=value
    q0[names.index('left_shoulder_roll_joint')]=.16;q0[names.index('right_shoulder_roll_joint')]=-.16
    # Ground distance uses all collision-mesh hull vertices, exact for a plane.
    vertices=[];vertex_links=[]
    for k,name in enumerate(robot.links.names):
        mesh=pk.collision.RobotCollision._get_trimesh_collision_geometries(u,name)
        v=np.asarray(mesh.vertices)
        if not len(v):continue
        try:v=v[ConvexHull(v).vertices]
        except Exception:pass
        vertices.extend(v);vertex_links.extend([k]*len(v))
    vertices=jnp.array(np.array(vertices),dtype=jnp.float32);vertex_links=jnp.array(vertex_links)
    fk0=jaxlie.SE3(robot.forward_kinematics(cfg=jnp.array(q0)))
    local_vertices=jaxlie.SE3(fk0.wxyz_xyz[vertex_links]).apply(vertices)
    root0=np.array([1.,0.,0.,0.,0.,0.,.84],dtype=np.float32)
    # Calibrate the sole to the ground in both directions: the legacy default
    # root height starts roughly6cm above support and is not a contact target.
    root0[6]=.001-float(local_vertices[:,2].min())
    initial_pose=jaxlie.SE3(jnp.array(root0))@fk0
    foot_goal=initial_pose.translation()[feet]
    foot_rotation=jaxlie.SO3(initial_pose.rotation().wxyz[feet])
    wrist0=np.asarray(initial_pose.rotation().wxyz)[wrist]
    desired=np.array([2**-.5,0.,2**-.5,0.])
    orientations=Slerp([0,1.5],Rotation.from_quat(np.array([wrist0,desired])[:,[1,2,3,0]]))(
        np.minimum(np.arange(n)/fps,1.5)).as_quat()[:,[3,0,1,2]]
    joint_vars=robot.joint_var_cls(jnp.arange(n));root_vars=jaxls.SE3Var(jnp.arange(n))

    @jaxls.Cost.create_factory
    def fit_cost(vals,q,r,decoded):
        pose=vals[r]@jaxlie.SE3(robot.forward_kinematics(cfg=vals[q]))
        return ((pose.translation()[links]-decoded)*3.).flatten()

    @jaxls.Cost.create_factory
    def hand_cost(vals,q,r,position,orientation):
        full=vals[r]@jaxlie.SE3(robot.forward_kinematics(cfg=vals[q]))
        pose=jaxlie.SE3(full.wxyz_xyz[wrist])
        return jnp.concatenate([(pose.translation()-position)*100.,
            (jaxlie.SO3(orientation).inverse()@pose.rotation()).log()*30.])

    @jaxls.Cost.create_factory
    def stance_cost(vals,q,r):
        full=vals[r]@jaxlie.SE3(robot.forward_kinematics(cfg=vals[q]))
        pose=jaxlie.SE3(full.wxyz_xyz[feet])
        return jnp.concatenate([((pose.translation()-foot_goal)*80.).flatten(),
            ((foot_rotation.inverse()@pose.rotation()).log()*30.).flatten()])

    @jaxls.Cost.create_factory
    def root_cost(vals,r):
        return (jaxlie.SE3(jnp.array(root0)).inverse()@vals[r]).log()*100.

    @jaxls.Cost.create_factory
    def table_cost(vals,q,r):
        pose=vals[r]@jaxlie.SE3(robot.forward_kinematics(cfg=vals[q]))
        world=jaxlie.SE3(pose.wxyz_xyz[vertex_links]).apply(vertices)
        delta=jnp.abs(world-jnp.array([.45,0.,.4]))-jnp.array([.3,.3,.4])
        distance=jnp.linalg.norm(jnp.maximum(delta,0)+1e-9,axis=-1)+jnp.minimum(jnp.max(delta,axis=-1),0)
        # Vertex-distance collision penalty; native PhysX replay remains the
        # authority because a vertex test alone cannot certify triangle clearance.
        return jnp.maximum(.01-jnp.min(distance),0)[None]*100.

    @jaxls.Cost.create_factory
    def ground_cost(vals,q,r):
        pose=vals[r]@jaxlie.SE3(robot.forward_kinematics(cfg=vals[q]))
        z=jaxlie.SE3(pose.wxyz_xyz[vertex_links]).apply(vertices)[:,2]
        # Minimum over every hull vertex is exact for the plane. A scalar active
        # constraint avoids materializing one 35-column Jacobian row per vertex.
        return jnp.maximum(.001-jnp.min(z),0)[None]*80.

    costs=[fit_cost(joint_vars,root_vars,jnp.array(target,dtype=jnp.float32)),
        hand_cost(joint_vars,root_vars,jnp.array(goals[:,21],dtype=jnp.float32),jnp.array(orientations,dtype=jnp.float32)),
        stance_cost(joint_vars,root_vars),root_cost(root_vars),ground_cost(joint_vars,root_vars),table_cost(joint_vars,root_vars),
        pk.costs.limit_cost(jax.tree.map(lambda x:x[None],robot),joint_vars,args.limit_weight),
        pk.costs.rest_cost(joint_vars,jnp.array(q0)[None],jnp.full((1,29),.1).at[:,12:15].set(3.)),
        pk.costs.smoothness_cost(robot.joint_var_cls(jnp.arange(1,n)),robot.joint_var_cls(jnp.arange(n-1)),jnp.array([2.]))]
    initial_q=jnp.broadcast_to(jnp.array(q0),(n,29));initial_root=jnp.broadcast_to(jnp.array(root0),(n,7))
    if args.initial_state:
        with np.load(args.initial_state,allow_pickle=False) as z:
            assert z['q'].shape==(n,29) and z['joint_names'].tolist()==names
            initial_q=jnp.array(z['q']);initial_root=jnp.array(np.concatenate([z['root_quat'],z['root_pos']],axis=1))
    initial=jaxls.VarValues.make([joint_vars.with_value(initial_q),root_vars.with_value(jaxlie.SE3(initial_root))])
    began=time.monotonic()
    solved=jaxls.LeastSquaresProblem(costs,[joint_vars,root_vars]).analyze().solve(initial_vals=initial)
    q=np.asarray(solved[joint_vars]);root=np.asarray(solved[root_vars].wxyz_xyz)
    poses=jax.vmap(lambda qq,rr:jaxlie.SE3(rr)@jaxlie.SE3(robot.forward_kinematics(cfg=qq)))(jnp.array(q),jnp.array(root))
    positions=np.asarray(poses.translation());quats=np.asarray(poses.rotation().wxyz)
    mesh_z=np.asarray(jaxlie.SE3(poses.wxyz_xyz[:,vertex_links]).apply(vertices[None]))[...,2]
    foot_speed=np.linalg.norm(np.diff(positions[:,np.asarray(feet),:2],axis=0),axis=-1)*fps
    sole_height=np.stack([mesh_z[:,np.asarray(vertex_links)==int(k)].min(axis=1) for k in np.asarray(feet)],axis=1)
    wrist_error=np.linalg.norm(positions[:,wrist]-goals[:,21],axis=-1)
    wrist_angle=(Rotation.from_quat(quats[:,wrist][:,[1,2,3,0]]).inv()*Rotation.from_quat(orientations[:,[1,2,3,0]])).magnitude()
    limits={j.get('name'):j.find('limit') for j in ET.parse(urdf).getroot().findall('joint') if j.get('type')=='revolute'}
    low=np.array([float(limits[k].get('lower')) for k in names]);high=np.array([float(limits[k].get('upper')) for k in names])
    violation=float(max(0,np.max(low-q),np.max(q-high)))
    checks={'finite':bool(np.isfinite(q).all() and np.isfinite(root).all()),'joint_limits':violation<=1e-4,
        'hand_target':bool(wrist_error.max()<.15),'wrist_orientation':bool(wrist_angle.max()<.35),
        'stance_foot_speed':bool(np.mean(foot_speed>.15)<=.1),'ground_mesh_plane':bool(mesh_z.min()>=-.005),
        'feet_on_ground':bool(np.all(sole_height<.02)),
        'root_height':bool(np.all((root[:,6]>.3)&(root[:,6]<1.2)))}
    args.output_dir.mkdir(exist_ok=False)
    path=args.output_dir/'robot29.npz'
    hand_closed=np.zeros((n,2),dtype=np.float32);hand_closed[np.arange(n)/fps>=2.,1]=1
    with path.open('xb') as f:np.savez_compressed(f,q=q,root_pos=root[:,4:],root_quat=root[:,:4],
        keypoints=positions[:,np.asarray(links)],right_wrist_quat=quats[:,wrist],hand_closed=hand_closed,
        joint_names=np.array(names),fps=np.array(20.),sample_times=np.arange(n)/fps)
    report={'status':'postoptimized_candidate','sample_sha256':sha(args.sample),'output':str(path),'output_sha256':sha(path),
        'urdf_sha256':sha(urdf),'scene_sha256':metadata['scene_sha256'],'frames':n,'joint_count':29,
        'max_hand_error_m':float(wrist_error.max()),'hand_error_at_grasp_m':float(wrist_error[40]),
        'max_wrist_angle_rad':float(wrist_angle.max()),'minimum_collision_mesh_height_m':float(mesh_z.min()),
        'maximum_foot_sole_height_m':float(sole_height.max()),'calibrated_initial_root_height_m':float(root0[6]),
        'max_joint_limit_violation_rad':violation,'max_stance_foot_speed_mps':float(foot_speed.max()),
        'checks':checks,'passed_requested_kinematic_checks':all(checks.values()),
        'native_scene_collision_replay':'pending','eligible_for_calibrated_scene_reference':False,
        'elapsed_s':time.monotonic()-began,'no_source_dataset_filtering':True,
        'joint_limit_objective_weight':args.limit_weight,
        'initial_state_sha256':sha(args.initial_state) if args.initial_state else None,
        'scope':'project postoptimizer with scene hand/stance constraints; no native table/hand collision or PPO success claimed',
        'capsule_self_collision_note':'not a physical gate: neutral logo/head fixed-link capsules overlap by0.14006m'}
    with (args.output_dir/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

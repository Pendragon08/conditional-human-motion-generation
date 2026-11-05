"""Fit two-wrist/rotating-wrist generated references with the compiled29DoF solver.

Only generated motions are processed. Source datasets and old baselines remain
read-only. CPU FK/mesh checks are recomputed after explicit tiny limit projection;
native scene replay, PD tracking and task success remain separate.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import pyroki as pk
import yourdfpy

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'stage3_pick29_20261103'
sys.path[:0]=[str(S/'src'),str(R/'training_disk_guard_20261101')]
from stage3_pick29_core import BODY_JOINTS
from stage3_supported_task_conditions import TASKS
from g1_fk import URDFFK, PAPER_22_LINKS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TaskSolver:
    def __init__(self):
        self.urdf=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
        assert sha(self.urdf)=='572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95'
        u=yourdfpy.URDF.load(str(self.urdf),load_meshes=False,load_collision_meshes=True)
        robot=pk.Robot.from_urdf(u);names=list(robot.joints.actuated_names)
        assert names==BODY_JOINTS
        self.robot,self.names=robot,names
        links=jnp.array([robot.links.names.index(k) for k in PAPER_22_LINKS])
        feet=jnp.array([robot.links.names.index(k+'_ankle_roll_link') for k in ('left','right')])
        wrist=jnp.array([robot.links.names.index(side+'_wrist_yaw_link') for side in ('left','right')])
        q0=np.zeros(29,dtype=np.float32)
        for side in ('left','right'):
            for part,value in [('hip_pitch',-.2),('knee',.42),('ankle_pitch',-.23),('elbow',.87),('shoulder_pitch',.35)]:
                q0[names.index(side+'_'+part+'_joint')]=value
        q0[names.index('left_shoulder_roll_joint')]=.16;q0[names.index('right_shoulder_roll_joint')]=-.16
        vertices=[];vertex_links=[];self.cpu_meshes={}
        for k,name in enumerate(robot.links.names):
            mesh=pk.collision.RobotCollision._get_trimesh_collision_geometries(u,name)
            all_v=np.asarray(mesh.vertices)
            if not len(all_v):continue
            self.cpu_meshes[name]=all_v.copy()
            try:v=all_v[ConvexHull(all_v).vertices]
            except Exception:v=all_v
            vertices.extend(v);vertex_links.extend([k]*len(v))
        vertices=jnp.array(np.asarray(vertices),dtype=jnp.float32);vertex_links=jnp.array(vertex_links)
        fk0=jaxlie.SE3(robot.forward_kinematics(cfg=jnp.array(q0)))
        local=jaxlie.SE3(fk0.wxyz_xyz[vertex_links]).apply(vertices)
        root0=np.array([1.,0.,0.,0.,0.,0.,.001-float(local[:,2].min())],dtype=np.float32)
        initial_pose=jaxlie.SE3(jnp.array(root0))@fk0
        foot_goal=initial_pose.translation()[feet]
        foot_rotation=jaxlie.SO3(initial_pose.rotation().wxyz[feet])
        wrist0=np.asarray(initial_pose.rotation().wxyz)[wrist]
        self.q0,self.root0=q0,root0
        self.joint_vars=robot.joint_var_cls(jnp.arange(180));self.root_vars=jaxls.SE3Var(jnp.arange(180))

        @jaxls.Cost.create_factory
        def fit_cost(vals,q,r,decoded):
            pose=vals[r]@jaxlie.SE3(robot.forward_kinematics(cfg=vals[q]))
            return ((pose.translation()[links]-decoded)*3.).flatten()

        @jaxls.Cost.create_factory
        def hand_cost(vals,q,r,position,orientation,hand_mask):
            full=vals[r]@jaxlie.SE3(robot.forward_kinematics(cfg=vals[q]))
            pose=jaxlie.SE3(full.wxyz_xyz[wrist])
            return jnp.concatenate([((pose.translation()-position)*hand_mask[:,None]*100.).flatten(),
                ((jaxlie.SO3(orientation).inverse()@pose.rotation()).log()*hand_mask[:,None]*30.).flatten()])

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
        def ground_cost(vals,q,r):
            pose=vals[r]@jaxlie.SE3(robot.forward_kinematics(cfg=vals[q]))
            z=jaxlie.SE3(pose.wxyz_xyz[vertex_links]).apply(vertices)[:,2]
            return jnp.maximum(.001-jnp.min(z),0)[None]*80.

        @jaxls.Cost.create_factory
        def table_cost(vals,q,r):
            pose=vals[r]@jaxlie.SE3(robot.forward_kinematics(cfg=vals[q]))
            world=jaxlie.SE3(pose.wxyz_xyz[vertex_links]).apply(vertices)
            delta=jnp.abs(world-jnp.array([.45,0.,.4]))-jnp.array([.3,.3,.4])
            distance=jnp.linalg.norm(jnp.maximum(delta,0)+1e-9,axis=-1)+jnp.minimum(jnp.max(delta,axis=-1),0)
            return jnp.maximum(.01-jnp.min(distance),0)[None]*100.

        self.costs=(fit_cost,hand_cost,stance_cost,root_cost,ground_cost,table_cost)
        self.fk=URDFFK(self.urdf);self.all_links=sorted(self.fk.links)
        self.kp=[self.all_links.index(k) for k in PAPER_22_LINKS]
        self.feet=[self.all_links.index(k+'_ankle_roll_link') for k in ('left','right')]
        self.wrist=[self.all_links.index(side+'_wrist_yaw_link') for side in ('left','right')]
        limits={j.get('name'):j.find('limit') for j in ET.parse(self.urdf).getroot().findall('joint') if j.get('type')=='revolute'}
        self.low=np.array([float(limits[k].get('lower')) for k in names])
        self.high=np.array([float(limits[k].get('upper')) for k in names])

    @partial(jax.jit,static_argnums=(0,))
    def solve(self,target,goals,orientation,hand_mask,initial_q,initial_root):
        robot=self.robot;q,r=self.joint_vars,self.root_vars
        fit,hand,stance,root,ground,table=self.costs
        costs=[fit(q,r,target),hand(q,r,goals,orientation,hand_mask),stance(q,r),root(r),ground(q,r),table(q,r),
            pk.costs.limit_cost(jax.tree.map(lambda x:x[None],robot),q,10000.),
            pk.costs.rest_cost(q,jnp.array(self.q0)[None],jnp.full((1,29),.1).at[:,12:15].set(3.)),
            pk.costs.smoothness_cost(robot.joint_var_cls(jnp.arange(1,180)),robot.joint_var_cls(jnp.arange(179)),jnp.array([2.]))]
        initial=jaxls.VarValues.make([q.with_value(initial_q),r.with_value(jaxlie.SE3(initial_root))])
        solved=jaxls.LeastSquaresProblem(costs,[q,r]).analyze().solve(initial_vals=initial)
        return solved[q],solved[r].wxyz_xyz

    def process(self,sample,output,initial_q,initial_root,initial_sha):
        output.mkdir(exist_ok=False)
        with np.load(sample,allow_pickle=False) as z:
            decoded=z['canonical_points'][0];goal=z['target_points'];m=json.loads(z['metadata_utf8'].tobytes())
            orientations=z['desired_wrist_quat_wxyz'].copy();closed=z['planned_hand_closed'].astype(np.float32);planned_object=z['desired_object_positions'].copy()
        assert decoded.shape==(180,22,3) and goal.shape==(180,22,3) and m['fps']==20 and m['task'] in TASKS
        rotation=np.array(m['canonical_to_world_rotation']);translation=np.array(m['canonical_to_world_translation'])
        target=decoded@rotation.T+translation;goals=goal@rotation.T+translation
        obj=np.array(m['task_object_position'])
        assert orientations.shape==(180,2,4) and closed.shape==(180,2)
        assert m['task_conditions_source_sha256']==sha(Path(__file__).with_name('stage3_supported_task_conditions.py'))
        hand_mask=np.ones((180,2),dtype=np.float32)
        if m['task']=='pour':hand_mask[:,0]=0
        if m['task']=='pour':assert np.allclose(goals[40,21],obj+[0,0,.04],atol=2e-7)
        began=time.monotonic()
        raw_q,raw_root=self.solve(jnp.array(target,dtype=jnp.float32),jnp.array(goals[:,20:22],dtype=jnp.float32),
                                  jnp.array(orientations,dtype=jnp.float32),jnp.array(hand_mask),jnp.array(initial_q),jnp.array(initial_root))
        raw_q=np.asarray(raw_q);raw_root=np.asarray(raw_root)
        solve_seconds=time.monotonic()-began
        with (output/'solver_raw.npz').open('xb') as f:np.savez_compressed(f,q=raw_q,root=raw_root)
        def reject(reason,metrics):
            result={'status':'generated_candidate_rejected','candidate_id':m['candidate_id'],
                'sample':str(sample),'sample_sha256':sha(sample),'solver_raw_sha256':sha(output/'solver_raw.npz'),
                'initial_state_sha256':initial_sha,'scene_sha256':m['scene_sha256'],'task':m['task'],'joint_count':29,'frames':180,
                'task_object_position':obj.tolist(),'rejection_reason':reason,'rejection_metrics':metrics,
                'passed_requested_kinematic_checks':False,'eligible_for_calibrated_scene_reference':False,
                'checks':{'candidate_admission':False},'solver_seconds':solve_seconds,
                'native_scene_collision_replay':'not_eligible','no_reference_admission_or_training':True}
            with (output/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
            return result
        if not (np.isfinite(raw_q).all() and np.isfinite(raw_root).all()):
            return reject('nonfinite_raw_solver_state',{'finite':False})
        q=np.clip(raw_q.astype(float),self.low+1e-6,self.high-1e-6).astype(np.float32)
        correction=float(np.max(np.abs(q-raw_q)))
        if correction>=.001:
            return reject('bound_correction_at_least1mrad',{'required_projection_rad':correction,'maximum_allowed_exclusive_rad':.001})
        points,rot=self.fk.forward(raw_root[:,4:],raw_root[:,[1,2,3,0]],q,self.names,links=self.all_links,return_rotations=True)
        minima=[];foot_heights=[]
        for name,v in self.cpu_meshes.items():
            k=self.all_links.index(name)
            z=(np.einsum('tij,vj->tvi',rot[:,k],v)+points[:,k,None])[:,:,2].min(axis=1)
            minima.append(z)
            if k in self.feet:foot_heights.append(z)
        minimum=float(np.stack(minima,1).min());sole_max=float(np.stack(foot_heights,1).max())
        speed=np.linalg.norm(np.diff(points[:,self.feet,:2],axis=0),axis=-1)*20
        errors=np.linalg.norm(points[:,self.wrist]-goals[:,20:22],axis=-1)
        wrist_q=Rotation.from_matrix(rot[:,self.wrist].reshape(-1,3,3)).as_quat()[:,[3,0,1,2]].reshape(180,2,4)
        angles=(Rotation.from_quat(wrist_q.reshape(-1,4)[:,[1,2,3,0]]).inv()*Rotation.from_quat(orientations.reshape(-1,4)[:,[1,2,3,0]])).magnitude().reshape(180,2)
        error=errors[hand_mask.astype(bool)];angle=angles[hand_mask.astype(bool)]
        grasp_error=float(errors[40][hand_mask[40].astype(bool)].max())
        violation=float(max(0,np.max(self.low-q),np.max(q-self.high)))
        checks={'finite':bool(np.isfinite(points).all()),'joint_limits':violation<1e-7,
            'hand_target':bool(error.max()<.03),'wrist_orientation':bool(angle.max()<.20),
            'stance_foot_speed':bool(np.mean(speed>.15)<=.1),'ground_mesh_plane':minimum>=-.005,
            'feet_on_ground':sole_max<.02,'root_height':bool(np.all((raw_root[:,6]>.3)&(raw_root[:,6]<1.2)))}
        assert np.isfinite(closed).all() and set(np.unique(closed)).issubset({0.,1.})
        path=output/'robot29.npz'
        with path.open('xb') as f:np.savez_compressed(f,q=q,root_pos=raw_root[:,4:],root_quat=raw_root[:,:4],
            keypoints=points[:,self.kp].astype(np.float32),left_wrist_quat=wrist_q[:,0].astype(np.float32),right_wrist_quat=wrist_q[:,1].astype(np.float32),hand_closed=closed,
            planned_object_positions=planned_object,desired_wrist_quat_wxyz=orientations,
            joint_names=np.array(self.names),fps=np.array(20.),sample_times=np.arange(180)/20.,
            task_object_position=obj,task_object_quaternion_wxyz=np.array([1.,0.,0.,0.]),
            metadata_utf8=np.frombuffer(json.dumps(m).encode(),dtype=np.uint8))
        result={'status':'postoptimized_bounded_cpu_recomputed','candidate_id':m['candidate_id'],
            'sample':str(sample),'sample_sha256':sha(sample),'output':str(path),'output_sha256':sha(path),
            'solver_raw_sha256':sha(output/'solver_raw.npz'),'initial_state_sha256':initial_sha,
            'source_initial_noise_sha256':m['initial_noise_sha256'],'source_feature_content_sha256':m['feature_content_sha256'],
            'urdf_sha256':sha(self.urdf),'scene_sha256':m['scene_sha256'],'task':m['task'],'frames':180,'joint_count':29,
            'task_object_position':obj.tolist(),'task_object_quaternion_wxyz':[1.,0.,0.,0.],
            'max_joint_projection_rad':correction,'max_joint_limit_violation_rad':violation,
            'max_hand_error_m':float(error.max()),'hand_error_at_grasp_m':grasp_error,
            'max_wrist_angle_rad':float(angle.max()),'minimum_collision_mesh_height_m':minimum,
            'maximum_foot_sole_height_m':sole_max,'max_stance_foot_speed_mps':float(speed.max()),
            'checks':checks,'passed_requested_kinematic_checks':all(checks.values()),
            'solver_seconds':solve_seconds,'elapsed_seconds':time.monotonic()-began,
            'native_scene_collision_replay':'pending','eligible_for_calibrated_scene_reference':False,
            'scope':'generated-only29DoF postoptimization, explicit tiny limit projection and NumPy FK/full collision-mesh plane checks; no task success'}
        with (output/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--max-candidates',type=int,default=4)
    p.add_argument('--follow',action='store_true',help='consume complete producer rows until its final report')
    p.add_argument('--exclude-records',type=Path,action='append',default=[])
    p.add_argument('--exclude-ids',default='')
    a=p.parse_args();assert 1<=a.max_candidates<=500
    a.output_dir.mkdir(exist_ok=False)
    (a.output_dir/'status.json').open('x').write(json.dumps({'status':'initializing_compiled_solver',
        'completed':0,'native_admitted':0,'input_records':str(a.records)},indent=2))
    assert any(d.platform=='gpu' for d in jax.devices()),'GPU solver required'
    solver=TaskSolver()
    initial_q=np.broadcast_to(solver.q0,(180,29)).copy();initial_root=np.broadcast_to(solver.root0,(180,7)).copy()
    initial_sha=hashlib.sha256(initial_q.tobytes()+initial_root.tobytes()).hexdigest()
    results=[]
    seen={int(x) for x in a.exclude_ids.split(',') if x}
    for previous in a.exclude_records:
        rows=[json.loads(x) for x in previous.read_text().splitlines()]
        if seen.intersection({r['candidate_id'] for r in rows}):raise ValueError('overlapping preserved prefixes')
        seen.update(r['candidate_id'] for r in rows)
    excluded=sorted(seen)
    waiting_started=time.monotonic()
    while len(results)<a.max_candidates:
        lines=a.records.read_text().splitlines(keepends=True) if a.records.exists() else []
        records=[json.loads(x) for x in lines if x.endswith('\n')]
        assert len({r['candidate_id'] for r in records})==len(records),'duplicate producer ID'
        for row in records:
            if row['candidate_id'] in seen:continue
            if len(results)>=a.max_candidates:break
            assert shutil.disk_usage('/data').free>8*1024**3+456*1024**2,'postopt disk protection'
            sample=Path(row['path']);assert sha(sample)==row['sha256']
            report=solver.process(sample,a.output_dir/('candidate_'+str(row['candidate_id']).zfill(6)),
                                  initial_q,initial_root,initial_sha)
            results.append(report);seen.add(row['candidate_id'])
            with (a.output_dir/'records.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
            status={'status':'processing','completed':len(results),'kinematic_passed':sum(r['passed_requested_kinematic_checks'] for r in results),
                    'native_admitted':0,'results':[r['candidate_id'] for r in results],'excluded_preserved_ids':excluded,
                    'rejected_candidates':[r['candidate_id'] for r in results if r['status']=='generated_candidate_rejected']}
            temp=a.output_dir/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(a.output_dir/'status.json')
        final=a.records.parent/'report.json'
        if final.exists():
            producer=json.loads(final.read_text());assert producer['status']=='sampling_completed_not_postoptimized'
            if len(results)<a.max_candidates:raise RuntimeError('producer completed before requested count')
        if not a.follow or len(results)>=a.max_candidates:break
        assert time.monotonic()-waiting_started<24000,'bounded producer wait expired'
        assert shutil.disk_usage('/data').free>8*1024**3+456*1024**2,'waiting disk protection'
        time.sleep(3)
    assert results
    status.update(status='postopt_completed_native_pending',solver_seconds=[r['solver_seconds'] for r in results],
                  input_records_sha256=sha(a.records))
    with (a.output_dir/'report.json').open('x') as f:json.dump(status,f,indent=2)
    (a.output_dir/'status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status),flush=True)


if __name__=='__main__':main()

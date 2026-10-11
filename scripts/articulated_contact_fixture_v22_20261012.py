"""Known articulated foot support, slip and air controls, no human policy."""
import argparse
import hashlib
import json
from pathlib import Path
from isaacgym import gymapi, gymtorch
import numpy as np
from physics_guided_metric_v21_20261012 import surface_metrics
from native_contact_diagnostic import read_contacts

ROOT=Path(__file__).resolve().parents[1]


def main(out):
    out.mkdir(exist_ok=False)
    urdf='''<robot name="known_articulated_foot">
<link name="base"><inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
<link name="foot"><inertial><origin xyz="0.03 0 0"/><mass value="5"/><inertia ixx="0.020833333" ixy="0" ixz="0" iyy="0.020833333" iyz="0" izz="0.033333333"/></inertial><collision><geometry><box size="0.2 0.2 0.1"/></geometry></collision><visual><geometry><box size="0.2 0.2 0.1"/></geometry></visual></link>
<link name="carrier"><inertial><mass value="0.1"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
<joint name="slider_x" type="prismatic"><parent link="base"/><child link="carrier"/><origin xyz="0 0 -0.95"/><axis xyz="1 0 0"/><limit lower="-10" upper="10" effort="10000" velocity="1"/></joint>
<joint name="support_z" type="prismatic"><parent link="carrier"/><child link="foot"/><axis xyz="0 0 1"/><limit lower="-0.03" upper="0.03" effort="10000" velocity="1"/></joint>
</robot>'''
    (out/'known_foot.urdf').write_text(urdf)
    gym=gymapi.acquire_gym();params=gymapi.SimParams()
    params.dt=1/60;params.substeps=2;params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81)
    params.use_gpu_pipeline=False;params.physx.use_gpu=True
    params.physx.contact_collection=gymapi.ContactCollection.CC_ALL_SUBSTEPS
    sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,params);assert sim is not None
    try:
        ground=gymapi.PlaneParams();ground.normal=gymapi.Vec3(0,0,1)
        ground.static_friction=0;ground.dynamic_friction=0;gym.add_ground(sim,ground)
        options=gymapi.AssetOptions();options.fix_base_link=True;options.collapse_fixed_joints=False
        options.linear_damping=0;options.angular_damping=0
        asset=gym.load_asset(sim,str(out),'known_foot.urdf',options)
        assert gym.get_asset_rigid_body_names(asset)==['base','carrier','foot']
        envs=[];centers=[]
        for i,(height,speed) in enumerate(((1.,0.),(1.,.2),(2.,0.))):
            env=gym.create_env(sim,gymapi.Vec3(-1,-1,0),gymapi.Vec3(1,1,3),2);envs.append(env)
            pose=gymapi.Transform();pose.p=gymapi.Vec3(0,0,height)
            actor=gym.create_actor(env,asset,pose,'known_foot_'+str(i),i,0)
            shapes=gym.get_actor_rigid_shape_properties(env,actor)
            for shape in shapes:shape.friction=0;shape.restitution=0
            gym.set_actor_rigid_shape_properties(env,actor,shapes)
            props=gym.get_actor_dof_properties(env,actor)
            props['driveMode'].fill(gymapi.DOF_MODE_NONE);props['stiffness'].fill(0);props['damping'].fill(0)
            props['driveMode'][0]=gymapi.DOF_MODE_VEL;props['damping'][0]=1000
            gym.set_actor_dof_properties(env,actor,props)
            initial=np.zeros(2,dtype=gymapi.DofState.dtype);initial['vel'][0]=speed
            gym.set_actor_dof_states(env,actor,initial,gymapi.STATE_ALL)
            gym.set_actor_dof_velocity_targets(env,actor,np.asarray([speed,0.],dtype=np.float32))
            com=gym.get_actor_rigid_body_properties(env,actor)[2].com
            centers.append([float(com.x),float(com.y),float(com.z)])
        gym.prepare_sim(sim)
        state=gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)).view(3,3,13)
        force=gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim)).view(3,3,3)
        states=[];forces=[];records=[];errors=[];dtypes=[]
        for k in range(120):
            gym.simulate(sim);gym.fetch_results(sim,True)
            gym.refresh_rigid_body_state_tensor(sim);gym.refresh_net_contact_force_tensor(sim)
            states.append(state.numpy().copy());forces.append(force.numpy().copy())
            read_contacts(gym,envs,k,records,errors,dtypes)
        states=np.stack(states);forces=np.stack(forces)
        assert np.isfinite(states).all() and np.isfinite(forces).all()
        thresholds=json.loads((ROOT/'results/physics_guided_20261011/frozen_thresholds.json').read_text())['thresholds']
        rows=[]
        points=[[x,y,z] for x in (-.1,.1) for y in (-.1,.1) for z in (-.05,.05)]
        for i in range(3):
            s=states[60:,i,2:3];geometry={'foot':dict(points=points,radii=[0]*8,com_local=centers[i])}
            rows.append(surface_metrics(s[...,:3],s[...,3:7],s[...,7:10],s[...,10:13],forces[60:,i,2:3],['foot'],geometry,thresholds))
        ground_counts=[sum(r['environment']==i and min(r['body0'],r['body1'])==-1 and r['force_magnitude_N']>0 for r in records) for i in range(3)]
        # Free vertical prismatic DOF allows the plane to bear foot weight.
        checks=dict(native_parser_no_errors=not errors,native_static_ground_contact=ground_counts[0]>0,
            native_sliding_ground_contact=ground_counts[1]>0,native_airborne_no_ground_contact=ground_counts[2]==0,
            static_surface_proxy_not_sliding=rows[0]['support_observations']>0 and rows[0]['surface_slip_proxy_fraction']==0,
            controlled_slider_surface_proxy_sliding=rows[1]['support_observations']>0 and rows[1]['surface_slip_proxy_fraction']==1,
            airborne_has_no_support=rows[2]['support_observations']==0,
            commanded_slide_measured_within_2cm_s=bool(np.max(np.abs(states[60:,1,2,7]-.2))<.02))
        np.savez_compressed(out/'fixture_arrays.npz',state=states,net_contact_force=forces,com_local=np.asarray(centers))
        result=dict(status='PASS' if all(checks.values()) else 'FAIL_RETAINED',checks=checks,rows=rows,
            body_names=['base','carrier','foot'],ground_counts=ground_counts,native_errors=errors,native_dtype_fields=dtypes,
            com_local=centers,thresholds_changed=False,innovation_confirmed=False,
            scope='Known translational articulated foot; does not certify arbitrary rolling contacts or real forces',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            arrays_sha256=hashlib.sha256((out/'fixture_arrays.npz').read_bytes()).hexdigest())
        (out/'summary.json').write_text(json.dumps(result,indent=2))
        print('ARTICULATED_CONTACT_FIXTURE',result['status'],json.dumps(checks),flush=True)
    finally:
        gym.destroy_sim(sim)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)

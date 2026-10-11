"""Independent rotating body / nonzero COM calibration, no human data."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from isaacgym import gymapi, gymtorch
import numpy as np
from physics_guided_core import rotate_xyzw


def main(out):
    out.mkdir(exist_ok=False)
    gym=gymapi.acquire_gym(); params=gymapi.SimParams()
    params.dt=1/240; params.substeps=2; params.up_axis=gymapi.UP_AXIS_Z
    params.gravity=gymapi.Vec3(0,0,0); params.use_gpu_pipeline=False;params.physx.use_gpu=True
    sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,params);assert sim is not None
    started=time.monotonic()
    try:
        options=gymapi.AssetOptions(); options.linear_damping=0;options.angular_damping=0
        asset=gym.create_box(sim,.2,.2,.2,options)
        centers=[]
        for i,cx in enumerate((0.,.06)):
            env=gym.create_env(sim,gymapi.Vec3(-1,-1,0),gymapi.Vec3(1,1,3),2)
            pose=gymapi.Transform();pose.p=gymapi.Vec3(0,0,2)
            actor=gym.create_actor(env,asset,pose,'rotating_offset_'+str(i),i,0)
            props=gym.get_actor_rigid_body_properties(env,actor)
            props[0].com=gymapi.Vec3(cx,0,0)
            gym.set_actor_rigid_body_properties(env,actor,props,recomputeInertia=False)
            checked=gym.get_actor_rigid_body_properties(env,actor)[0].com
            centers.append([float(checked.x),float(checked.y),float(checked.z)])
            assert np.max(np.abs(np.asarray(centers[-1])-[cx,0,0]))<1e-6
            state=gym.get_actor_rigid_body_states(env,actor,gymapi.STATE_ALL)
            state['vel']['linear']['x']=.2;state['vel']['angular']['z']=3.
            gym.set_actor_rigid_body_states(env,actor,state,gymapi.STATE_ALL)
        gym.prepare_sim(sim)
        bodies=gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
        assert tuple(bodies.shape)==(2,13) and bodies.device.type=='cpu'
        rows=[]
        for step in range(240):
            gym.simulate(sim);gym.fetch_results(sim,True);gym.refresh_rigid_body_state_tensor(sim)
            rows.append(bodies.numpy().copy())
        state=np.stack(rows);assert np.isfinite(state).all()
        lever=rotate_xyzw(state[:,:,3:7],np.asarray(centers)[None])
        origin=np.gradient(state[:,:,:3].astype(float),params.dt,axis=0)
        com=np.gradient(state[:,:,:3].astype(float)+lever,params.dt,axis=0)
        original_residual=np.linalg.norm(origin-state[:,:,7:10],axis=-1)[2:-2]
        com_residual=np.linalg.norm(com-state[:,:,7:10],axis=-1)[2:-2]
        a=float(np.max(original_residual[:,1]));b=float(np.max(com_residual[:,1]))
        origin_supported=a<.003 and b>.05
        com_supported=b<.003 and a>.05
        checks=dict(zero_offset_origin_error_below_3mm_s=float(np.max(original_residual[:,0]))<.003,
                    angular_rate_constant=bool(np.max(np.abs(state[:,:,12]-3.))<.001),
                    exactly_one_velocity_reference_supported=bool(origin_supported != com_supported))
        np.savez_compressed(out/'fixture_arrays.npz',state=state,com_local=np.asarray(centers))
        result=dict(status='PASS' if all(checks.values()) else 'FAIL_RETAINED',checks=checks,
            exported_linear_velocity_reference='BODY_ORIGIN' if origin_supported else 'CENTER_OF_MASS' if com_supported else 'UNRESOLVED',
            maximum_origin_hypothesis_error_m_s=a,maximum_com_hypothesis_error_m_s=b,
            dt=params.dt,steps=240,com_local=centers,gravity=[0,0,0],angular_velocity_rad_s=3.,
            thresholds_preregistered=True,humanoid_metric_replaced=False,innovation_confirmed=False,
            scope='Native rigid-body velocity reference on rotating isolated body, not articulated contact-slip calibration',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            arrays_sha256=hashlib.sha256((out/'fixture_arrays.npz').read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-started)
        (out/'summary.json').write_text(json.dumps(result,indent=2))
        print('VELOCITY_REFERENCE_FIXTURE',result['status'],result['exported_linear_velocity_reference'],flush=True)
    finally:
        gym.destroy_sim(sim)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)

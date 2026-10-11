"""Actual PhysX box/plane contact calibration; no controller or generator."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from isaacgym import gymapi, gymtorch
import numpy as np
import torch


def main(out):
    out.mkdir(exist_ok=False)
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1/60
    params.substeps = 2
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.use_gpu_pipeline = False
    params.physx.use_gpu = True
    params.physx.contact_collection = gymapi.ContactCollection.CC_ALL_SUBSTEPS
    params.physx.num_position_iterations = 4
    params.physx.num_velocity_iterations = 0
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    assert sim is not None
    started = time.monotonic()
    try:
        ground = gymapi.PlaneParams()
        ground.normal = gymapi.Vec3(0, 0, 1)
        ground.static_friction = 0
        ground.dynamic_friction = 0
        ground.restitution = 0
        gym.add_ground(sim, ground)
        options = gymapi.AssetOptions()
        options.density = 1250.0  # 0.2*0.2*0.1m box -> 5kg.
        options.linear_damping = 0
        options.angular_damping = 0
        asset = gym.create_box(sim, .2, .2, .1, options)
        sensor = gymapi.ForceSensorProperties()
        sensor.enable_forward_dynamics_forces = False
        sensor.enable_constraint_solver_forces = True
        sensor.use_world_frame = True
        sensor_index = gym.create_asset_force_sensor(asset, 0, gymapi.Transform(), sensor)
        assert sensor_index == 0
        fixtures = [('stationary_grounded', .05, 0., 0.),
                    ('frictionless_slide', .05, .2, 0.),
                    ('high_airborne_drop', 2., 0., 0.),
                    ('upward_takeoff', .3, 0., 2.)]
        masses = []
        for i, (name, height, vx, vz) in enumerate(fixtures):
            env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 3), 2)
            pose = gymapi.Transform(); pose.p = gymapi.Vec3(0, 0, height)
            actor = gym.create_actor(env, asset, pose, name, i, 0)
            shapes = gym.get_actor_rigid_shape_properties(env, actor)
            for shape in shapes:
                shape.friction = 0
                shape.restitution = 0
            gym.set_actor_rigid_shape_properties(env, actor, shapes)
            state = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
            state['vel']['linear']['x'] = vx
            state['vel']['linear']['z'] = vz
            gym.set_actor_rigid_body_states(env, actor, state, gymapi.STATE_ALL)
            masses.append(float(gym.get_actor_rigid_body_properties(env, actor)[0].mass))
        gym.prepare_sim(sim)
        root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        contact = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
        sensor_tensor = gymtorch.wrap_tensor(gym.acquire_force_sensor_tensor(sim))
        assert tuple(root.shape) == (4, 13)
        assert tuple(contact.shape) == (4, 3)
        assert tuple(sensor_tensor.shape) == (4, 6)
        states = []; forces = []; readings = []
        for step in range(120):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_actor_root_state_tensor(sim)
            gym.refresh_net_contact_force_tensor(sim)
            gym.refresh_force_sensor_tensor(sim)
            torch.cuda.synchronize()
            states.append(root.detach().cpu().numpy().copy())
            forces.append(contact.detach().cpu().numpy().copy())
            readings.append(sensor_tensor.detach().cpu().numpy().copy())
        states = np.stack(states); forces = np.stack(forces); readings = np.stack(readings)
        assert all(np.isfinite(x).all() for x in (states, forces, readings))
        weights = np.asarray(masses)*9.81
        checks = {
            'frictionless_horizontal_net_force_below_1N': bool(np.max(np.abs(forces[:,:,:2])) < 1),
            'settled_stationary_height_within_1cm': bool(np.max(np.abs(states[-30:, 0, 2]-.05)) < .01),
            'settled_stationary_speed_below_2cm_s': bool(np.max(np.linalg.norm(states[-30:, 0, 7:10], axis=-1)) < .02),
            'net_force_weight_relative_error_below_10pct': bool(np.max(np.abs(forces[-30:, 0, 2]-weights[0]))/weights[0] < .10),
            'sensor_force_weight_relative_error_below_10pct': bool(np.max(np.abs(readings[-30:, 0, 2]-weights[0]))/weights[0] < .10),
            'sliding_velocity_retained_within_2cm_s': bool(np.max(np.abs(states[-30:, 1, 7]-.2)) < .02),
            'high_airborne_initial_force_below_1N': bool(np.max(np.abs(forces[:20, 2])) < 1),
            'high_airborne_initial_sensor_below_1N': bool(np.max(np.abs(readings[:20, 2, :3])) < 1),
            'high_airborne_initial_height_above_ground': bool(np.min(states[:20, 2, 2]) > .5),
        }
        np.savez_compressed(out/'fixture_arrays.npz', state=states, net_contact_force=forces, constraint_force_sensor=readings)
        metadata = dict(status='PASS' if all(checks.values()) else 'FAIL_RETAINED', checks=checks,
            fixtures=[x[0] for x in fixtures], masses_kg=masses, weights_N=weights.tolist(),
            dt=params.dt, steps=120, substeps=2, contact_collection=2,
            generator_used=False, controller_used=False, thresholds_preregistered=True,
            export_buffer_cleared_before_each_simulate=False,
            use_gpu_pipeline=False, physx_use_gpu=True,
            clearing_scope='No clearing; CPU tensor export with unchanged GPU PhysX solver',
            simulator_force_sensor_is_not_real_force_ground_truth=True,
            interpretation='Known box/plane physics fixture; does not certify humanoid contact points, articulation sensors or slip proxy.',
            stationary_net_force_range_N=[float(forces[-30:, 0, 2].min()), float(forces[-30:, 0, 2].max())],
            stationary_sensor_force_range_N=[float(readings[-30:, 0, 2].min()), float(readings[-30:, 0, 2].max())],
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            arrays_sha256=hashlib.sha256((out/'fixture_arrays.npz').read_bytes()).hexdigest(),
            elapsed_seconds=time.monotonic()-started)
        (out/'summary.json').write_text(json.dumps(metadata, indent=2))
        print('CONTACT_BOX_FIXTURE', metadata['status'], json.dumps(checks), flush=True)
    finally:
        torch.cuda.synchronize()
        gym.destroy_sim(sim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)

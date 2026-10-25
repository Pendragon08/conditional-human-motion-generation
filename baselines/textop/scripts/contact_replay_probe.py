"""Replay three saved first-step outliers; counterfactuals are diagnosis only.

No learned policy, optimizer, dataset filtering, or shared dependency changes.
The first recorded action is held fixed to isolate physical execution.
"""
import argparse, hashlib, json, os, pathlib, time, traceback
from isaaclab.app import AppLauncher

ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p = argparse.ArgumentParser()
p.add_argument('--spec', required=True)
p.add_argument('--replay-spec', required=True)
p.add_argument('--case', required=True, choices=['baseline', 'no_self_collision', 'no_floor'])
p.add_argument('--output', required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
out = pathlib.Path(a.output)
out.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
receipt = {'status': 'STARTING', 'case': a.case, 'optimizer_updates': 0,
           'new_checkpoints_written': 0, 'dataset_accepted': False,
           'physical_filter_executed': False, 'formal_filter_training_started': False,
           'full_generator_training_started': False, 'output_writers_closed': False}
app = env = None

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def save():
    receipt['elapsed_seconds'] = time.monotonic() - started
    partial = out / 'receipt.json.partial'
    partial.write_text(json.dumps(receipt, indent=2, allow_nan=False))
    partial.replace(out / 'receipt.json')

try:
    spec = json.loads(pathlib.Path(a.spec).read_text())
    replay = json.loads(pathlib.Path(a.replay_spec).read_text())
    assert a.headless and a.device == 'cuda:1' and os.getuid() == 1013
    assert set(os.sched_getaffinity(0)) <= {0, 1, 2, 3}
    assert hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:8] == replay['warm_environment_canonical_hash']
    assert sha(spec['robot_urdf']) == spec['robot_urdf_sha256']
    for rel, expected in replay['pinned_sources'].items():
        assert sha(ROOT / rel) == expected, rel
    private = ROOT / replay['private_input_directory']
    for name, expected in replay['private_input_sha256'].items():
        assert sha(private / name) == expected
    alerts = json.loads((private / 'first_state_alert.private.json').read_text())
    assert len(alerts) == replay['num_envs'] == 3
    assert all(sha(x['source']['file']) == x['source']['sha256'] for x in alerts)
    receipt.update(source_sha256=sha(__file__), spec_sha256=sha(a.replay_spec),
                   private_input_sha256=replay['private_input_sha256'], num_envs=3,
                   cpu_affinity=sorted(os.sched_getaffinity(0)),
                   diagnostic_only_changes=replay['cases'][a.case],
                   action_protocol='Recorded first action held constant; no policy inference or optimization')
    save()
    app = AppLauncher(a).app
    import torch, gymnasium as gym, numpy as np
    from paper_filter_config import make
    torch.set_num_threads(4)
    torch.manual_seed(replay['seed'])
    torch.cuda.manual_seed_all(replay['seed'])
    torch.cuda.set_per_process_memory_fraction(replay['torch_memory_fraction'], torch.device(a.device))
    assert torch.cuda.mem_get_info(a.device)[0] > replay['minimum_free_gpu_gib'] * 1024**3
    initial = torch.load(private / 'initial_state.private.pth', map_location='cpu', weights_only=True)
    original_ids = torch.tensor([x['environment'] for x in alerts], dtype=torch.long)
    root = initial['root_state_w'][original_ids].to(a.device)
    jp = initial['joint_pos'][original_ids].to(a.device)
    jv = initial['joint_vel'][original_ids].to(a.device)
    actions = torch.tensor([x['action'] for x in alerts], dtype=torch.float32, device=a.device)
    cfg, _ = make(spec, [x['source']['file'] for x in alerts], out / 'usd', phase='relaxed', num_envs=3)
    if a.case == 'no_self_collision':
        cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    if a.case == 'no_floor':
        cfg.scene.terrain = None
    env = gym.make('Tracking-Flat-G1-PrivPrivObs-v0', cfg=cfg)
    raw = env.unwrapped
    env.reset()
    robot = raw.scene['robot']
    # Use exact saved WORLD coordinates; no inferred origin translation.
    # Small-scene layout/contact warm starts need not be bitwise equivalent to
    # the original8192 scene, so explicitly compare the first executed state.
    robot.write_joint_state_to_sim(jp, jv)
    robot.write_root_state_to_sim(root)
    raw.sim.forward()
    raw.scene.update(0.0)
    assert robot.data.joint_pos.shape == (3, 29)
    assert torch.allclose(robot.data.root_state_w, root, atol=1e-5, rtol=0)
    assert torch.allclose(robot.data.joint_pos, jp, atol=1e-6, rtol=0)
    assert torch.allclose(robot.data.joint_vel, jv, atol=1e-6, rtol=0)

    def stats(tensor):
        tensor = tensor.detach()
        finite = torch.isfinite(tensor)
        valid = tensor[finite]
        return {'min': float(valid.min()) if valid.numel() else None,
                'max': float(valid.max()) if valid.numel() else None,
                'abs_max': float(valid.abs().max()) if valid.numel() else None,
                'nonfinite': int((~finite).sum())}

    masses = robot.root_physx_view.get_masses()
    inertia = robot.root_physx_view.get_inertias().reshape(3, -1, 3, 3)
    eig = torch.linalg.eigvalsh(inertia)
    receipt.update(body_mass=stats(masses), inertia_eigenvalues=stats(eig),
                   nonpositive_inertia_eigenvalues=int((eig <= 0).sum()),
                   joint_names=robot.joint_names, body_names=robot.body_names,
                   initial_joint_velocity=stats(robot.data.joint_vel),
                   initial_root_height=stats(robot.data.root_pos_w[:, 2]),
                   initial_body_origin_height=stats(robot.data.body_pos_w[:, :, 2]),
                   action=stats(actions), simulation_dt=raw.physics_dt,
                   first_step_equivalence_not_assumed=True)
    private_out = out / 'replay_private'
    private_out.mkdir()
    np.savez(private_out / 'initial.private.npz', root_state=root.cpu().numpy(),
             joint_pos=jp.cpu().numpy(), joint_vel=jv.cpu().numpy(),
             body_state=robot.data.body_state_w.cpu().numpy(), actions=actions.cpu().numpy())
    raw.action_manager.process_action(actions)
    steps = []
    save()
    with torch.inference_mode():
        for index in range(replay['physics_substeps']):
            raw.action_manager.apply_action()
            raw.scene.write_data_to_sim()
            raw.sim.step(render=False)
            raw.scene.update(raw.physics_dt)
            forces = raw.scene['contact_forces'].data.net_forces_w
            item = {'physics_step': index + 1,
                    'joint_velocity': stats(robot.data.joint_vel),
                    'body_linear_velocity': stats(robot.data.body_lin_vel_w),
                    'body_angular_velocity': stats(robot.data.body_ang_vel_w),
                    'contact_force_components': stats(forces),
                    'body_origin_height': stats(robot.data.body_pos_w[:, :, 2]),
                    'applied_torque': stats(robot.data.applied_torque)}
            if index == 0:
                expected_jv = torch.tensor([x['joint_vel'] for x in alerts], device=a.device)
                expected_root = torch.tensor([x['root_state_w'] for x in alerts], device=a.device)
                receipt['first_step_difference_from_original'] = {
                    'joint_velocity': stats(robot.data.joint_vel - expected_jv),
                    'root_state': stats(robot.data.root_state_w - expected_root)}
                np.savez(private_out / 'first_step.private.npz', root_state=robot.data.root_state_w.cpu().numpy(),
                         joint_pos=robot.data.joint_pos.cpu().numpy(), joint_vel=robot.data.joint_vel.cpu().numpy(),
                         body_state=robot.data.body_state_w.cpu().numpy(), contact=forces.cpu().numpy())
            steps.append(item)
            receipt['steps'] = steps
            save()
            assert all(v['nonfinite'] == 0 for v in item.values() if isinstance(v, dict)), 'Nonfinite executed state; preserve and stop diagnostic'
    receipt.update(status='TEXTOP_CONTACT_REPLAY_DIAGNOSTIC_COMPLETED', output_writers_closed=True,
                   physics_substeps_completed=len(steps), peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
                   limitation='Three-state controlled contact diagnostic; not full-library stability, trained tracking or physical filtering')
    save()
    print(receipt['status'], flush=True)
except BaseException as error:
    receipt.update(status='FAILED', error=repr(error), traceback=traceback.format_exc())
    save()
    raise
finally:
    if app is not None:
        from bounded_headless_close import close
        close(app, receipt, save, out, simulation=env.unwrapped.sim if env is not None else None,
              successful_statuses=('TEXTOP_CONTACT_REPLAY_DIAGNOSTIC_COMPLETED',))

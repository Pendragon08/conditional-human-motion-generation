"""Derive a separate Pick evaluator from the already validated frozen-policy bridge."""
from pathlib import Path
import sys

root, out = map(Path, sys.argv[1:3])
evaluator = root / 'evaluate_native2000_tasks_v2_20261105.py'
source = evaluator.read_text(encoding='utf-8-sig')


def replace(old, new):
    global source
    assert source.count(old) == 1, old
    source = source.replace(old, new, 1)


replace("import stage3_pick29_env as base", """import stage3_pick29_env as base
    import isaaclab.sim as sim_utils
    from isaaclab.managers import SceneEntityCfg
    from upstream_pick_adapter_20261105 import configure, REVISION, BASE, PICK as UPSTREAM_PICK, PLAY""")
replace("device, task, count = args.device, args.task, args.num_envs",
        "device, task, count = args.device, args.task, args.num_envs\n    assert task == 'pick'")
replace("scene = {'object_size': base.SCENE['object_size']}", """scene = {'object_size': base.SCENE['object_size']}
        upstream_reset, upstream_score, shift_xyz = configure(
            cfg, base.SCENE['table_center'], sim_utils, isaac_mdp, EventTermCfg, SceneEntityCfg, ManagerBasedRLEnv, torch)
        shift = torch.tensor(shift_xyz, device=device)
        object_positions = object_positions + shift
        reset_z = float(upstream_reset.params['height'])""")
replace("seen_contact = torch.zeros_like(first)", """seen_contact = torch.zeros_like(first)
    max_closed_z = torch.full((count,), -torch.inf, device=device)""")
replace("env._task_cache_step = -1\n\n            def reset", """env._task_cache_step = -1
                env.n_successes = torch.zeros(count, device=device)
                env.motion_ids = torch.arange(count, device=device)
                env.start_motion_times = torch.zeros(count, device=device)

                class MotionBank:
                    def get_motion_state(self, ids, times):
                        frames = torch.floor(times * 20 + 1e-4).long().clamp(0, hands.shape[1] - 1)
                        return {'grab_pos': object_positions[ids],
                                'offsets': torch.zeros_like(object_positions[ids]),
                                'is_closed': hands[ids, frames, 1] > .5}

                env.motion_lib = MotionBank()

            def reset""")
replace("x[:, :3] + env.scene.env_origins[env_ids]", "x[:, :3] + shift + env.scene.env_origins[env_ids]")
replace("""                obj = env.scene['object']
                state = obj.data.default_root_state[env_ids].clone()
                state[:, :3] = object_positions[env_ids] + env.scene.env_origins[env_ids]
                state[:, 3:7] = object_quats[env_ids]
                state[:, 7:] = 0
                obj.write_root_state_to_sim(state, env_ids=env_ids)""",
        "                upstream_reset.func(env, env_ids, **upstream_reset.params)")
replace("""                    z = env.scene['object'].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
                    passed = (z > base.SCENE['success_height']) & (steps >= target_step)""",
        """                    old = env.n_successes.clone()
                    upstream_score(env, height_thres=.95, fall_thres=.9)
                    passed = env.n_successes > old
                    closed = env.motion_lib.get_motion_state(env.motion_ids, steps * env.step_dt)['is_closed']
                    z = env.scene['object'].data.root_pos_w[:, 2]
                    assert torch.equal(passed, (z > .95) & closed)
                    max_closed_z.copy_(torch.where(first & closed, torch.maximum(max_closed_z, z), max_closed_z))""")
replace("assert torch.allclose(pos, object_positions, atol=2e-5, rtol=0.)", """expected_pos = object_positions.clone()
            expected_pos[:, 2] = reset_z
            assert torch.allclose(pos, expected_pos, atol=2e-5, rtol=0.)
            actual_pos = pos.cpu().tolist()
            mass = sim_env.scene['object'].root_physx_view.get_masses().reshape(-1)
            assert mass.shape == (count,) and bool(((mass >= .1 - 1e-6) & (mass <= .5 + 1e-6)).all())
            masses = mass.cpu().tolist()
            assert len(set(masses)) > 1""")
replace("root[:, :3] - sim_env.scene.env_origins, root[:, 3:]",
        "root[:, :3] - sim_env.scene.env_origins - shift, root[:, 3:]")
replace("'status': 'frozen_policy_task_evaluation_completed', 'task': task, 'episode_count': count,",
        """'status': 'frozen_policy_task_evaluation_completed', 'task': task, 'episode_count': count,
        'upstream_commit': REVISION,
        'upstream_sources': {p.name: sha(p) for p in (BASE, UPSTREAM_PICK, PLAY)},
        'upstream_scene': {'table_center': list(cfg.scene.table.init_state.pos),
                           'table_size': list(cfg.scene.table.spawn.size),
                           'object_size': list(cfg.scene.object.spawn.size),
                           'object_reset_height': reset_z, 'object_mass_add_uniform_kg': [0., .4]},
        'frame_alignment_xyz_m': shift_xyz,
        'scope_limit': 'Public Pick tabletop, mass event, reset and counter; project 29-DoF robot, frame-zero reset and paired reference positions; first reference episode only. Wrist error is a project diagnostic.',""")
replace("'actual object center above 0.95 m at or after the grasp step (height criterion)'",
        "'upstream object_above_threshold counter increases when object center z>0.95 m and reference is_closed, first episode'")
replace("'object_initial_xyz_m': positions[i]",
        """'object_initial_xyz_m': actual_pos[i],
             'object_mass_kg': masses[i],
             'max_closed_object_height_m': float(max_closed_z[i]) if bool(torch.isfinite(max_closed_z[i])) else None""")

compile(source, 'evaluate_upstream_pick.py', 'exec')
(out / 'evaluate_upstream_pick.py').write_text(source, encoding='utf-8')
controller = (root / 'run_native2000_task_evaluation_v2_20261105.py').read_text(encoding='utf-8-sig')
controller = controller.replace('from evaluate_native2000_tasks_v2_20261105 import', 'from evaluate_upstream_pick import')
controller = controller.replace("'evaluate_native2000_tasks_v2_20261105.py'", "'evaluate_upstream_pick.py'")
controller = controller.replace("    child = None\n    try:", """    source_paths += [Path(__file__).with_name(name) for name in (
        'upstream_pick_adapter_20261105.py', 'motion_tracking_interactive_base.py',
        'motion_tracking_pick_env.py', 'play_eval.py')]
    child = None
    try:""", 1)
compile(controller, 'run_upstream_pick.py', 'exec')
(out / 'run_upstream_pick.py').write_text(controller, encoding='utf-8')

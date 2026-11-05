"""Online SuperMimic-style adaptation of the same 29DoF PhysX Pick task.

Real transitions -> autoregressive world model -> differentiable imagined
rollout/policy update -> critic fitting. No PPO update is used. Imagined
transitions do not consume the explicitly matched simulator-interaction budget.
"""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
import io
import hashlib

R = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S = R/'stage3_pick29_20261103'
D = R/'stage3_supermimic29_20261103'
sys.path.insert(0, str(S/'src'))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--num-envs', type=int, default=8)
p.add_argument('--macro-rounds', type=int, default=4)
p.add_argument('--warmup-steps', type=int, default=64)
p.add_argument('--model-warmup-updates', type=int, default=4)
p.add_argument('--replay-rows', type=int, default=32)
p.add_argument('--model-batch', type=int, default=64)
p.add_argument('--policy-batch', type=int, default=64)
p.add_argument('--imagination-horizon', type=int, default=32)
p.add_argument('--stop-after-vector-steps', type=int)
p.add_argument('--resume', type=Path)
p.add_argument('--formal', action='store_true')
p.add_argument('--capacity-test', action='store_true')
p.add_argument('--diagnostic', action='store_true')
p.add_argument('--reference-manifest', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.formal:
    assert (args.num_envs, args.macro_rounds, args.warmup_steps, args.model_warmup_updates) == (8912, 3000, 64, 500)
    assert args.stop_after_vector_steps is None
    assert shutil.disk_usage('/data').free >= 11*1024**3, 'formal startup requires11GiB free'
    assert not args.capacity_test
elif args.capacity_test:
    assert args.num_envs == 8912 and args.macro_rounds == 4
elif args.diagnostic:
    assert args.num_envs == 8912 and args.macro_rounds == 25 and args.model_warmup_updates == 500
else:
    assert 2 <= args.num_envs <= 16 and 3 <= args.macro_rounds <= 8
assert args.replay_rows == 32 and 8 <= args.model_batch <= 128 and 8 <= args.policy_batch <= 128
assert args.imagination_horizon == 32 and args.warmup_steps >= 32
args.output_dir.mkdir(exist_ok=False, parents=True)
run = args.output_dir
report = {'status': 'initializing', 'pid': os.getpid(), 'formal_run': args.formal,
          'algorithm': 'SuperMimic-style world model and BPTT adaptation, not PPO',
          'task_success_claimed': False}
started = time.monotonic()
env = app = None


def status(**values):
    report.update(values)
    temporary = run/'status.tmp'
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(run/'status.json')


try:
    status()
    app = AppLauncher(args).app
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    import stage3_pick29_env as base
    import stage3_pick500_env as paired
    from stage3_pick29_core import BODY_JOINTS, sha
    from stage3_resume_contract import equal_tree
    from stage3_supermimic_budget import RealInteractionBudget
    from stage3_supermimic_replay import RealReplay
    from stage3_supermimic_pick_core_v2 import (native_primitives, LivePolicy, LiveCritic, PickWorldModel,
        live_obs, hard_hands, private_reference, fit_world_model, improve_policy)
    torch.set_num_threads(2)
    torch.manual_seed(20261103); np.random.seed(20261103); random.seed(20261103)
    torch.cuda.set_per_process_memory_fraction(.08, device=args.device)
    plan = json.loads((D/'handoff_plan.json').read_text())
    native = native_primitives(plan['existing_textop_environment']['source_root'])
    physical_ids = None

    def state_of(e):
        robot, obj = e.scene['robot'], e.scene['object']
        return torch.cat((robot.data.root_pos_w-e.scene.env_origins, robot.data.root_quat_w,
            robot.data.root_lin_vel_w, robot.data.root_ang_vel_w,
            robot.data.joint_pos[:, physical_ids], robot.data.joint_vel[:, physical_ids],
            obj.data.root_pos_w-e.scene.env_origins, obj.data.root_quat_w,
            obj.data.root_lin_vel_w, obj.data.root_ang_vel_w), -1)

    def capture(e):
        # This reward hook runs before automatic episode reset. Return the exact
        # original reward value; capture terminal state/reference for replay.
        e._sm_next = state_of(e).clone()
        e._sm_reference = private_reference(base.reference(e)).clone()
        e._sm_terminated = base.fallen(e).clone()
        return mdp.is_alive(e)

    cfg = paired.Pick500EnvCfg()
    cfg.scene.num_envs = args.num_envs; cfg.sim.device = args.device; cfg.seed = 20261103
    cfg.reference_manifest = str(args.reference_manifest)
    cfg.rewards.alive.func = capture
    env = ManagerBasedRLEnv(cfg=cfg); wrapped = RslRlVecEnvWrapper(env)
    names = BODY_JOINTS+sorted(n for n in env.scene['robot'].joint_names if n not in BODY_JOINTS)
    assert len(names) == 43 and len(set(names)) == 43
    physical_ids = env.scene['robot'].find_joints(names, preserve_order=True)[0]
    default_q = env.scene['robot'].data.default_joint_pos[0, physical_ids].clone()
    default_qd = env.scene['robot'].data.default_joint_vel[0, physical_ids].clone()
    observation, _ = wrapped.reset()
    assert env._reference_bank.count == 500
    selected = env._reference_ids.clone()
    expected = env._reference_bank.object_positions[selected]+env.scene.env_origins
    object_error = float((env.scene['object'].data.root_pos_w-expected).abs().max())
    before = base.live_observation(env).clone()
    env._reference_ids = (selected+1)%500
    leak = float((base.live_observation(env)-before).abs().max())
    env._reference_ids = selected
    assert object_error < 1e-5 and leak == 0.
    status(reference_count=500, reference_manifest_sha256=env._reference_bank.manifest_sha256,
        object_reset_error_m=object_error, actor_reference_change_error=leak,
        physical_body_dof=29, physical_joint_count=43, actions=31, actor_observation=102,
        capacity_test=args.capacity_test, device=args.device)
    policy, critic, wm = LivePolicy(native).to(args.device), LiveCritic(native).to(args.device), PickWorldModel(native).to(args.device)
    optimizers = {
        'policy': torch.optim.Adam(policy.parameters(), lr=.001, betas=(.9, .95)),
        'critic': torch.optim.Adam(critic.parameters(), lr=.001),
        'world_model': torch.optim.Adam(wm.parameters(), lr=.0002),
    }
    models = {'policy': policy, 'critic': critic, 'world_model': wm}
    replay = RealReplay(args.replay_rows, args.num_envs, args.device)
    budget = RealInteractionBudget(args.num_envs, args.macro_rounds, 24)
    counters = {'world_model_updates': 0, 'policy_updates': 0, 'model_warmup_done': 0,
                'utd_tenths': 0, 'completed_episodes': 0}
    config = {k: getattr(args, k) for k in ['num_envs', 'macro_rounds', 'warmup_steps', 'model_warmup_updates',
        'replay_rows', 'model_batch', 'policy_batch', 'imagination_horizon', 'formal', 'capacity_test', 'diagnostic']}
    bindings = {'asset_sha256': sha(base.ASSET), 'reference_bank_sha256': env._reference_bank.manifest_sha256,
                'physical_joint_names': names, 'actor_observation': 102, 'actions': 31, 'actor_reference_input': False,
                'scene_sha256': base.SCENE_SHA, 'reference_count':500, 'object_pose_policy':'per_reference_fixed',
                'code_sha256': {f: sha(D/'src'/f) for f in ['stage3_supermimic_pick500_train_v2.py',
                    'stage3_supermimic_pick_core_v2.py', 'stage3_supermimic_replay.py', 'stage3_supermimic_budget.py']}}
    bindings['task_source_sha256'] = {f:sha(S/'src'/f) for f in ['stage3_pick500_env.py','stage3_pick29_env.py','stage3_pick29_core.py']}
    (run/'bindings.json').open('x').write(json.dumps(bindings,indent=2))
    if args.resume:
        restored = torch.load(args.resume, map_location=args.device, weights_only=True)
        assert restored['config'] == config and restored['bindings'] == bindings, 'resume configuration/source mismatch'
        for k, model in models.items(): model.load_state_dict(restored[k])
        for k, optimizer in optimizers.items(): optimizer.load_state_dict(restored[k+'_optimizer'])
        replay.load_state_dict(restored['replay'])
        budget = RealInteractionBudget.restore(restored['budget'], expected_num_envs=args.num_envs,
            expected_macro_rounds=args.macro_rounds, expected_steps_per_round=24)
        assert replay.total_rows == budget.vector_steps
        counters.update(restored['counters'])
        torch.set_rng_state(restored['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in restored['cuda_rng']])
        random.setstate(restored['python_rng'])
        nr = restored['numpy_rng']
        np.random.set_state((nr[0], nr[1].cpu().numpy().astype(np.uint32), nr[2], nr[3], nr[4]))
        # Physics is freshly initialized after restart; optimizer, replay, RNG
        # and total interaction budget resume. Do not claim bitwise continuity.
        status(resume_checkpoint=str(args.resume), resume_sha256=sha(args.resume),
               restored_vector_steps=budget.vector_steps, simulator_reinitialized=True)

    def full_checkpoint(label):
        path = run/('resume_'+label+'.pt')
        estimate = 160*1024**2+replay.bytes
        assert shutil.disk_usage('/data').free >= 8*1024**3+356*1024**2+estimate, 'checkpoint disk protection'
        nr = np.random.get_state()
        value = {k: model.state_dict() for k, model in models.items()}
        value.update({k+'_optimizer': v.state_dict() for k, v in optimizers.items()})
        value.update(config=config, bindings=bindings, counters=counters.copy(), budget=budget.state_dict(),
            replay=replay.state_dict(), default_q=default_q, default_qd=default_qd,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(), numpy_rng=(nr[0], torch.from_numpy(nr[1].astype(np.int64)), nr[2], nr[3], nr[4]))
        if args.capacity_test or args.diagnostic:
            buffer = io.BytesIO()
            torch.save(value, buffer)
            serialized_bytes = buffer.tell()
            digest = hashlib.sha256(buffer.getbuffer()).hexdigest()
            buffer.seek(0)
            check = torch.load(buffer, map_location='cpu', weights_only=True)
        else:
            with path.open('xb') as f: torch.save(value, f)
            check = torch.load(path, map_location='cpu', weights_only=True)
            serialized_bytes = path.stat().st_size
            digest = sha(path)
        verified = all(equal_tree(value[k], check[k]) for k in list(models)+[k+'_optimizer' for k in optimizers]+['budget', 'counters', 'replay'])
        assert verified, 'full checkpoint roundtrip mismatch'
        receipt = {'path': None if (args.capacity_test or args.diagnostic) else str(path), 'sha256': digest, 'bytes': serialized_bytes,
                   'in_memory_capacity_serialization':args.capacity_test or args.diagnostic,
                   'roundtrip': verified, 'vector_steps': budget.vector_steps}
        with (run/'checkpoints.jsonl').open('a') as f: f.write(json.dumps(receipt)+'\n')
        status(latest_checkpoint=receipt)
        return receipt

    status(status='online_training', config=config, bindings=bindings, replay_bytes=replay.bytes,
           progress=budget.progress(), counters=counters.copy(),
           adaptation_differences=['learned reward/termination head fitted to unchanged real PhysX rewards',
               'same live Actor instead of reference-conditioned native Actor',
               '112 physical state incl43 joints and object;31 action; bounded32-row replay',
               'binary hand straight-through gradient only within world model',
               'policy-only checkpoints50 rounds/full replay at96steps,1500rounds,final; emergency separate',
               'stable FP64 global gradient norm; reject nonfinite components instead of masking'])
    maximum_observation_error = 0.
    last_metrics = {}
    cycle_started = time.monotonic()
    stop_at = args.stop_after_vector_steps or budget.target_vector_steps
    assert budget.vector_steps < stop_at <= budget.target_vector_steps
    while budget.remaining_vector_steps and budget.vector_steps < stop_at:
        assert shutil.disk_usage('/data').free >= 8*1024**3+356*1024**2, 'runtime disk protection'
        if (run/'REQUEST_STOP').exists():
            full_checkpoint('protected_'+str(budget.vector_steps)); status(status='protected_stop'); break
        warmup = budget.vector_steps < args.warmup_steps
        with torch.no_grad():
            s = state_of(env); previous = env.action_manager.action.clone()
            error = float((live_obs(s, previous, default_q, default_qd)-observation).abs().max())
            maximum_observation_error = max(maximum_observation_error, error)
            assert error < 1e-4, 'Actor observation parity failure'
            policy.norm.update(observation); critic.norm.update(observation)
            action = hard_hands(policy(observation)+torch.randn_like(previous))
            steps = env.episode_length_buf.clone(); ids = env._reference_ids.clone()
            observation, reward, done, extras = wrapped.step(action)
            row = {'s': s, 's1': env._sm_next, 'a': action, 'a_prev': previous,
                'r': reward, 'term': env._sm_terminated, 'done': done.bool(),
                'ref': env._sm_reference, 'step': steps, 'id': ids}
            replay.append(row); budget.record_after_step(actual_num_envs=args.num_envs, warmup=warmup)
            wm.update_norms(row)
            counters['completed_episodes'] += int(done.sum())
        if budget.vector_steps >= args.warmup_steps:
            while counters['model_warmup_done'] < args.model_warmup_updates:
                last_metrics.update(fit_world_model(wm, optimizers['world_model'], replay.sample(args.model_batch, 8)))
                counters['model_warmup_done'] += 1; counters['world_model_updates'] += 1
                if counters['model_warmup_done'] % 50 == 0:
                    status(status='world_model_warmup', counters=counters.copy(), progress=budget.progress())
            if not warmup:
                # Exact integer accumulator: UTD0.4 => two/five updates per real vector step.
                counters['utd_tenths'] += 4
                if counters['utd_tenths'] >= 10:
                    counters['utd_tenths'] -= 10
                    last_metrics.update(fit_world_model(wm, optimizers['world_model'], replay.sample(args.model_batch, 8)))
                    counters['world_model_updates'] += 1
                    starts = {k: v[:, 0] for k, v in replay.sample(args.policy_batch, 1).items() if k in ['s', 'a_prev', 'step', 'id']}
                    last_metrics.update(improve_policy(policy, critic, wm, optimizers['policy'], optimizers['critic'],
                        starts, env._reference_bank, default_q, default_qd, horizon=args.imagination_horizon))
                    counters['policy_updates'] += 1
        if budget.vector_steps % 24 == 0:
            metric = {'elapsed_seconds': time.monotonic()-started, **budget.progress(), **counters, **last_metrics,
                'interval_seconds': time.monotonic()-cycle_started,
                'torch_allocated_peak_bytes': torch.cuda.max_memory_allocated(args.device)}
            with (run/'metrics.jsonl').open('a') as f: f.write(json.dumps(metric, allow_nan=False)+'\n')
            status(status='online_training', progress=budget.progress(), counters=counters.copy(),
                   metrics=metric, live_observation_max_error=maximum_observation_error)
            cycle_started = time.monotonic()
            round_number = budget.vector_steps//24
            if args.formal and budget.vector_steps==96:
                full_checkpoint('early_96')
            if round_number % 50 == 0 and budget.remaining_vector_steps:
                path = run/('policy_'+str(round_number).zfill(4)+'.pt')
                with path.open('xb') as f: torch.save({'policy': policy.state_dict(), 'bindings': bindings,
                    'budget': budget.state_dict(), 'default_q': default_q, 'default_qd': default_qd}, f)
            if round_number % 1500 == 0 and budget.remaining_vector_steps:
                full_checkpoint(str(round_number).zfill(4))
    if report['status'] != 'protected_stop':
        receipt = full_checkpoint('final' if not budget.remaining_vector_steps else 'partial_'+str(budget.vector_steps))
        status(status='interaction_budget_completed' if not budget.remaining_vector_steps else 'bounded_prefix_completed',
               progress=budget.progress(), counters=counters.copy(), elapsed_seconds=time.monotonic()-started,
               live_observation_max_error=maximum_observation_error,
               peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(args.device))
        print('SUPERMIMIC_ONLINE_RUN_COMPLETED '+json.dumps(report), flush=True)
except BaseException as error:
    # Preserve exact interaction counts even when the failing step lies between
    # the24-step periodic metric records. Invalid updates are never accepted.
    extra={}
    if 'budget' in globals():extra.update(progress=budget.progress(),counters=counters.copy())
    status(status='failed', error=repr(error), traceback=traceback.format_exc(),**extra)
    if args.formal and 'full_checkpoint' in globals():
        try:full_checkpoint('failure_'+str(budget.vector_steps))
        except BaseException as checkpoint_error:status(failure_checkpoint_error=repr(checkpoint_error))
    print(report['traceback'], flush=True)
    raise
finally:
    if env is not None: env.close()
    if app is not None: app.close(wait_for_replicator=False)

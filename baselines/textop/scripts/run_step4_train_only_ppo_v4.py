"""Fresh431D/557D one-stage PPO; explicit real SDK/recovery witnesses first."""
import argparse
import math
import os
import signal
import shutil
import time
import traceback
from isaaclab.app import AppLauncher
from step4_train_only_protocol import ROOT, RESERVE, LOG, read, sha, owned, atomic, validate_policy

p = argparse.ArgumentParser(); p.add_argument('--request', required=True)
AppLauncher.add_app_launcher_args(p); a = p.parse_args()
sp = owned(a.request); s = read(sp); policy = validate_policy(s['policy'])
assert os.getuid() == 1013 and a.headless and a.device == 'cuda:1'
assert set(os.sched_getaffinity(0)) == {0, 1, 2, 3} and sha(__file__) == s['executor_sha256']
for n, h in s['bound_sources'].items(): assert sha(owned(ROOT/n)) == h
from step4_train_only_ppo_inputs import accepted_library
accepted, rows = accepted_library(s['library_binding'])
diagnostic = s['role'] in ['author', 'document']
assert (diagnostic and policy['witnesses_authorized'] and not policy['training_authorized']) or (not diagnostic and policy['training_authorized'])
parent = owned(s['output_root']); out = parent/s['run']; out.mkdir(mode=0o700, exist_ok=False)
started = time.monotonic(); app = env = runner = None; optimizer_calls = 0; optimizer_attempts = 0; completed = 0; last_good = None
rec = dict(status='STARTING_STAGE4_ORIGINAL_TRAIN_ONE_STAGE_PPO', run=s['run'], role=s['role'],
    source_sha256=sha(__file__), request_sha256=sha(sp), input_library_binding=s['library_binding'],
    training_data_mode='original_train_only', stage3_status='POSTPONED_BY_USER', original_train_families=5361,
    actual_original_train_clips=5551, original_VAL_motions_in_training=0, generated_training_clips=0,
    actor_observation_dim=431, critic_observation_dim=557, action_dim=29, policy_hz=50,
    diagnostic_optimizer_calls=0, formal_optimizer_updates=0, formal_completed_iterations=0,
    actual_environment_steps=0, output_writers_closed=False, old_teacher_training_updates=0,
    VAE_updates=0, LDM_updates=0)


def save():
    rec['elapsed_seconds'] = time.monotonic()-started; atomic(out/'receipt.json', rec)


save()
def interrupted(signum, frame):
    raise RuntimeError('Owned Stage4 PPO interrupted with signal '+str(signum))
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    assert shutil.disk_usage(out).free >= RESERVE+LOG+s['output_upper_bytes']
    app = AppLauncher(a).app
    import random
    import numpy as np
    import torch
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from filter_training_runtime import RecordedMotionCommand
    from step4_train_only_isaaclab_config import make
    from step4_train_only_ppo_runtime_v4 import snapshot, restore, durable, content, finite, equal, replay_update
    torch.set_num_threads(4)
    torch.cuda.set_device(torch.device(a.device))
    random.seed(policy['seed']); np.random.seed(policy['seed']); torch.manual_seed(policy['seed']); torch.cuda.manual_seed(policy['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(False)
    torch.cuda.set_per_process_memory_fraction(.65, torch.device(a.device))
    assert torch.cuda.mem_get_info(a.device)[0] >= 20*1024**3
    environment = read(ROOT/'configs/tracker-env-spec.json')
    cfg, agent = make(policy, environment, accepted, [x['file'] for x in rows], out/'usd')
    cfg.commands.motion.class_type = RecordedMotionCommand
    env = RslRlVecEnvWrapper(gym.make('Tracking-Flat-G1-ProjGravObs-v0', cfg=cfg))
    motion = env.unwrapped.command_manager.get_term('motion')
    obs, extras = env.get_observations()
    assert tuple(obs.shape) == (8192, 431) and tuple(extras['observations']['critic'].shape) == (8192, 557)
    assert motion.num_motion == 5551 and env.num_actions == 29
    assert torch.isfinite(obs).all() and torch.isfinite(extras['observations']['critic']).all()
    assert int((motion.raw_failed+motion.raw_success).sum()) == 0
    # Separate policy initialization from SDK setup's random draws; the new
    # actor/critic uses the declared seed, never a diagnostic/teacher payload.
    torch.manual_seed(policy['seed']); torch.cuda.manual_seed(policy['seed'])
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(out/'training_logs'), device=a.device)
    assert runner.training_type == 'rl' and runner.alg.__class__.__name__ == 'PPO'
    # No pretrained checkpoint of any kind enters the new initialization.
    assert len(runner.alg.optimizer.state) == 0
    initial = snapshot(runner, motion, 0, 0, s['library_binding'])
    rec.update(status='RUNNING_STAGE4_REAL_PPO_RECOVERY_WITNESS' if diagnostic else 'RUNNING_STAGE4_FRESH_ORIGINAL_TRAIN_PPO',
        num_envs=8192, initialized_model_sha256=content(initial['model']),
        actual_actor_observation_shape=list(obs.shape), actual_critic_observation_shape=list(extras['observations']['critic'].shape),
        parameter_count=sum(p.numel() for p in runner.alg.policy.parameters()),
        initial_optimizer_state_entries=0, simulator_episode_restart_on_resume=True,
        bit_exact_PhysX_cache_replay_claimed=False)
    rec['runtime_numeric_sources'] = {}
    import inspect
    for key, cls in [('runner', OnPolicyRunner), ('ppo', runner.alg.__class__), ('actor', runner.alg.policy.__class__)]:
        path = inspect.getfile(cls); rec['runtime_numeric_sources'][key] = dict(path=path, sha256=sha(path))
    save(); last_good = initial; base_completed = 0
    if s['role'] == 'document':
        item = s['author_checkpoint']; checkpoint = owned(item['path'])
        assert checkpoint.parent == parent/'user_step4_train_only_ppo_author_v4'
        assert checkpoint.stat().st_size == item['bytes'] and sha(checkpoint) == item['sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['completed_iterations'] == 1 and state['optimizer_updates'] == 20
        assert content(state) == item['content_sha256']
        restore(runner, motion, state, s['library_binding'], restart_episodes=True)
        base_completed = 1; last_good = state
        rec.update(cross_process_author_checkpoint_restored=True, author_checkpoint_sha256=item['sha256'],
            author_restored_fullstate_content_sha256=content(state),
            source_completed_iterations=1, source_optimizer_updates=20)
        save()
    original_step = runner.alg.optimizer.step

    def counted(*args, **kwargs):
        global optimizer_calls, optimizer_attempts
        optimizer_attempts += 1
        rec['optimizer_attempts_in_process'] = optimizer_attempts
        assert all(torch.isfinite(x).all() and x.grad is not None and torch.isfinite(x.grad).all() for x in runner.alg.policy.parameters())
        result = original_step(*args, **kwargs)
        optimizer_calls += 1
        rec['actual_optimizer_calls_in_process'] = optimizer_calls
        if diagnostic: rec['diagnostic_optimizer_calls'] = optimizer_calls
        assert all(torch.isfinite(x).all() for x in runner.alg.policy.parameters())
        return result

    runner.alg.optimizer.step = counted
    original_env_step = env.step

    def observed_step(actions):
        assert tuple(actions.shape) == (8192, 29) and torch.isfinite(actions).all()
        result = original_env_step(actions)
        assert torch.isfinite(result[0]).all() and torch.isfinite(result[1]).all()
        rec['actual_environment_steps'] += 1
        return result

    env.step = observed_step
    original_log = runner.log
    native_update = runner.alg.update
    def original_update():
        # Keep strict exact deterministic NN math, without forcing unsupported
        # cumsum/multinomial kernels on native SDK episode sampling.
        prior = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try: return native_update()
        finally: torch.use_deterministic_algorithms(prior)
    replay = None
    if diagnostic:
        def replaying_update():
            global replay
            if replay is None and max(base_completed, completed) >= 1:
                retained_before = max(base_completed, completed)
                state = snapshot(runner, motion, retained_before, retained_before*20, s['library_binding'])
                runner.alg.update = original_update
                try: replay = replay_update(runner, motion, state, s['library_binding'])
                finally: runner.alg.update = replaying_update
            return original_update()
        runner.alg.update = replaying_update
    target = 2 if diagnostic else policy['target_RL_iterations']
    schedules = [1, 2] if diagnostic else [x for x in [10000, 50000, 100000, target] if x <= target]
    schedules = sorted(set(schedules))
    metrics = out/'metrics.private.jsonl'
    rec['checkpoints'] = []

    def monitored_log(values, *args, **kwargs):
        global completed, last_good
        completed = int(values['it'])+1
        if completed <= 10 or completed % 100 == 0 or completed == target:
            original_log(values, *args, **kwargs)
        else:
            # Preserve the runner clocks, bounding printed/Tensorboard logs;
            # every actual iteration still has a private scalar journal row.
            runner.tot_timesteps += runner.num_steps_per_env*env.num_envs*runner.gpu_world_size
            runner.tot_time += values['collection_time']+values['learn_time']
        calls_expected = (completed-base_completed)*20+(40 if diagnostic and replay is not None else 0)
        assert optimizer_calls == calls_expected
        losses = {k: float(v) for k, v in values['loss_dict'].items()}; assert all(math.isfinite(v) for v in losses.values())
        last_good = snapshot(runner, motion, completed, completed*20, s['library_binding'])
        assert finite(last_good)
        rec.update(diagnostic_optimizer_calls=optimizer_calls if diagnostic else 0,
            formal_optimizer_updates=0 if diagnostic else completed*20,
            formal_completed_iterations=0 if diagnostic else completed,
            diagnostic_logical_iterations=completed if diagnostic else 0,
            latest_losses=losses, latest_collection_seconds=float(values['collection_time']),
            latest_learning_seconds=float(values['learn_time']),
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
            raw_completed_attempts=int((motion.raw_failed+motion.raw_success).sum()))
        import json
        with metrics.open('a') as f:
            os.chmod(metrics, 0o600); f.write(json.dumps(dict(completed=completed, optimizer_calls=optimizer_calls,
                losses=losses, collection_seconds=values['collection_time'], learning_seconds=values['learn_time']))+'\n')
            f.flush(); os.fsync(f.fileno())
        remaining = len([x for x in schedules if x >= completed])+1
        assert shutil.disk_usage(out).free >= RESERVE+LOG+remaining*s['state_slot_bytes']
        if completed in schedules:
            item = durable(out/('completed_'+str(completed)+'.private.pth'), last_good, s['state_slot_bytes'])
            loaded = torch.load(item['path'], map_location='cpu', weights_only=False)
            restore(runner, motion, loaded, s['library_binding'])
            assert equal(snapshot(runner, motion, completed, completed*20, s['library_binding']), loaded)
            rec['checkpoints'].append(item)
        save()

    runner.log = monitored_log
    runner.save = lambda *args, **kwargs: None
    runner.learn(num_learning_iterations=target-base_completed, init_at_random_ep_len=False)
    assert completed == target
    if diagnostic:
        assert replay and replay['real_recorded_rollout_exact_replay'] and optimizer_calls == (target-base_completed)*20+40
        saved = last_good; restore(runner, motion, saved, s['library_binding'], restart_episodes=True)
        rec.update(status='STAGE4_TRAIN_ONLY_REAL_PPO_RECOVERY_WITNESS_FINISHED',
            recorded_rollout_replay=replay, exact_model_Adam_normalizers_all_RNG_ADS_restore_verified=True,
            formal_updates=0, diagnostic_weights_will_not_seed_formal_training=True,
            diagnostic_logical_iterations=2, diagnostic_new_logical_iterations=target-base_completed,
            diagnostic_optimizer_calls=optimizer_calls)
    else:
        assert optimizer_calls == target*20
        rec.update(status='STAGE4_ORIGINAL_TRAIN_ONE_STAGE_PPO_FINISHED', formal_completed_iterations=completed,
            formal_optimizer_updates=optimizer_calls)
    accepted_library(s['library_binding'])
    if s['role'] == 'document': assert sha(s['author_checkpoint']['path']) == s['author_checkpoint']['sha256']
    for n, h in s['bound_sources'].items(): assert sha(owned(ROOT/n)) == h
    if runner.writer is not None: runner.writer.flush(); runner.writer.close(); runner.writer = None
    rec.update(output_writers_closed=True, inputs_unchanged=True); save()
except BaseException as error:
    rec.update(status='STAGE4_ORIGINAL_TRAIN_PPO_FAILED_PARTIAL_RETAINED', error=repr(error), traceback=traceback.format_exc())
    if last_good is not None and 'durable' in globals():
        try:
            assert shutil.disk_usage(out).free >= RESERVE+LOG+s['state_slot_bytes']
            rec['emergency_last_good'] = durable(out/'emergency_last_good.private.pth', last_good, s['state_slot_bytes'])
        except BaseException as e: rec['emergency_save_error'] = repr(e)
    save(); raise
finally:
    if runner is not None and runner.writer is not None: runner.writer.flush(); runner.writer.close(); runner.writer = None
    if app is not None:
        from bounded_headless_close import close
        close(app, rec, save, out, simulation=env.unwrapped.sim if env is not None else None,
            successful_statuses=('STAGE4_TRAIN_ONLY_REAL_PPO_RECOVERY_WITNESS_FINISHED', 'STAGE4_ORIGINAL_TRAIN_ONE_STAGE_PPO_FINISHED'))

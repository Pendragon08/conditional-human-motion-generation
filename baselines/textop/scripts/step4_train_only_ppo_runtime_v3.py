"""Full new PPO state and exact recorded-rollout replay.

Physics episodes explicitly restart on process continuation. PhysX caches and
in-flight episodes are not claimed bit-exact. PPO/model/Adam/normalizers,
algorithm learning rate, all RNG streams and adaptive sampling statistics are.
"""
import copy
import hashlib
import io
import math
import os
import pathlib
import random
import numpy as np
import torch


def clone(x):
    if torch.is_tensor(x): return x.detach().cpu().clone()
    if isinstance(x, dict): return {k: clone(v) for k, v in x.items()}
    if isinstance(x, list): return [clone(v) for v in x]
    if isinstance(x, tuple): return tuple(clone(v) for v in x)
    return copy.deepcopy(x)


def equal(x, y):
    if torch.is_tensor(x): return torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x.cpu(), y.cpu())
    if isinstance(x, np.ndarray): return isinstance(y, np.ndarray) and x.dtype == y.dtype and np.array_equal(x, y)
    if isinstance(x, dict): return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)): return type(x) == type(y) and len(x) == len(y) and all(equal(a, b) for a, b in zip(x, y))
    return x == y


def content(x):
    h = hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            a = v.detach().cpu().contiguous(); h.update(('tensor'+str(a.dtype)+str(tuple(a.shape))).encode()); h.update(a.numpy().tobytes())
        elif isinstance(v, np.ndarray): h.update(('numpy'+str(v.dtype)+str(v.shape)).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=lambda k: (type(k).__name__, str(k))): add(k); add(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(type(v).__name__.encode())
            for z in v: add(z)
        else: h.update((type(v).__name__+':'+repr(v)+'\0').encode())
    add(x); return h.hexdigest()


def rng():
    # Every CUDA RNG stream used by this single-device job, without opening a
    # context on another user's GPU merely to inspect an unused generator.
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
        cuda_device1=torch.cuda.get_rng_state(torch.device('cuda:1')))


def restore_rng(x):
    random.setstate(x['python']); np.random.set_state(x['numpy']); torch.set_rng_state(x['cpu'])
    torch.cuda.set_rng_state(x['cuda_device1'], torch.device('cuda:1')); assert equal(rng(), x)


ADS_NAMES = ['failed_motion_count', 'success_motion_count', 'current_failed_motion_count', 'current_success_motion_count',
    'failed_total_count', 'success_total_count', 'raw_failed', 'raw_success']


def ads(motion): return {k: clone(getattr(motion, k)) for k in ADS_NAMES}


@torch.inference_mode()
def restore_ads(motion, state):
    for k, v in state.items():
        if torch.is_tensor(v): getattr(motion, k).copy_(v.to(motion.device))
        else: setattr(motion, k, v)
    assert equal(ads(motion), state)


def finite(x):
    if torch.is_tensor(x): return not x.is_floating_point() or bool(torch.isfinite(x).all())
    if isinstance(x, dict): return all(finite(v) for v in x.values())
    if isinstance(x, (list, tuple)): return all(finite(v) for v in x)
    return not isinstance(x, float) or math.isfinite(x)


def snapshot(runner, motion, completed, updates, binding):
    states = runner.alg.optimizer.state_dict()['state']
    if completed == 0:
        assert not states
    else:
        assert len(states) == len(list(runner.alg.policy.parameters()))
        assert all(int(v['step']) == updates for v in states.values())
    value = dict(schema='textop.stage4_train_only_fullstate.v1', completed_iterations=completed, optimizer_updates=updates,
        input_binding=binding, model=clone(runner.alg.policy.state_dict()), optimizer=clone(runner.alg.optimizer.state_dict()),
        obs_normalizer=clone(runner.obs_normalizer.state_dict()), critic_normalizer=clone(runner.privileged_obs_normalizer.state_dict()),
        learning_rate=runner.alg.learning_rate, rng=clone(rng()), adaptive_sampling=ads(motion),
        runner_clock=dict(tot_time=runner.tot_time, tot_timesteps=runner.tot_timesteps),
        physics_continuation='new episodes; simulator caches/in-flight episodes not serialized',
        algorithm='fresh one-stage PPO; no teacher/DAgger/distillation')
    assert finite(value) and updates == completed*20
    return value


def restore(runner, motion, value, binding, restart_episodes=False):
    assert value['schema'] == 'textop.stage4_train_only_fullstate.v1' and value['input_binding'] == binding
    assert finite(value) and value['optimizer_updates'] == value['completed_iterations']*20
    runner.alg.policy.load_state_dict(value['model'], strict=True); runner.alg.optimizer.load_state_dict(value['optimizer'])
    # RSL updates cached RMS _std inside its inference-mode rollout. Load
    # non-learned buffers in that same context, preserving exact values.
    # Adam/learned parameters remain restored outside inference mode.
    with torch.inference_mode():
        runner.obs_normalizer.load_state_dict(value['obs_normalizer'], strict=True)
        runner.privileged_obs_normalizer.load_state_dict(value['critic_normalizer'], strict=True)
    runner.alg.learning_rate = value['learning_rate']
    runner.current_learning_iteration = value['completed_iterations']
    runner.tot_time = value['runner_clock']['tot_time']; runner.tot_timesteps = value['runner_clock']['tot_timesteps']
    restore_ads(motion, value['adaptive_sampling']); restore_rng(value['rng'])
    actual = snapshot(runner, motion, value['completed_iterations'], value['optimizer_updates'], binding)
    assert equal(actual, value)
    if restart_episodes:
        # Exact state is verified before the explicitly new physics episodes.
        # Reset consumes a real new RNG continuation; do not silently rewind it.
        with torch.inference_mode(): motion._env.reset()
        restore_ads(motion, value['adaptive_sampling']); motion._attempt_started.fill_(True)
    return actual


def durable(path, state, upper):
    path = pathlib.Path(path); assert not path.exists()
    partial = path.with_name(path.name+'.partial'); assert not partial.exists()
    with partial.open('xb') as f:
        os.chmod(partial, 0o600); torch.save(state, f); f.flush(); os.fsync(f.fileno())
    assert partial.stat().st_size <= upper
    loaded = torch.load(partial, map_location='cpu', weights_only=False)
    assert equal(loaded, state) and finite(loaded)
    partial.rename(path)
    fd = os.open(str(path.parent), os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''): h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest(), content_sha256=content(state))


def replay_update(runner, motion, state, binding):
    """Two actual updates on one real SDK rollout; both begin at exact state.

    Storage includes actions/advantages/returns and real observed transitions.
    This verifies optimizer/minibatch RNG continuation, not simulator replay.
    """
    storage = copy.deepcopy(runner.alg.storage)
    assert storage.step == runner.num_steps_per_env
    restore(runner, motion, state, binding)
    first_losses = runner.alg.update()
    first = snapshot(runner, motion, state['completed_iterations']+1, state['optimizer_updates']+20, binding)
    restore(runner, motion, state, binding); runner.alg.storage = copy.deepcopy(storage)
    second_losses = runner.alg.update()
    second = snapshot(runner, motion, state['completed_iterations']+1, state['optimizer_updates']+20, binding)
    assert equal(first_losses, second_losses) and equal(first, second)
    restore(runner, motion, state, binding); runner.alg.storage = storage
    return dict(real_recorded_rollout_exact_replay=True, losses=first_losses, replay_content_sha256=content(first),
        restored_content_sha256=content(state), optimizer_calls=40, formal_updates=0)

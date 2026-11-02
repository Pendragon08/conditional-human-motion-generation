"""Local draft CPU audit/export core for a future completed mixed PPO state.

No launch path. Do not call until a trusted, separately reviewed supervisor
owns the common lock and has accepted all authentic completed-run evidence.
This module is not deployed, and its real-model path is not yet tested.
"""
import hashlib
import inspect
import math
import os
import pathlib
import random

from augmentation2526_mixed1000_fullstate_metadata_v1 import validate_mixed_fullstate_metadata

NATIVE_SHA = {
    'actor': '51ce8c6504a7434f86e5188ccb11e9fc981e46f764c00e32be3e8febe7112ae2',
    'normalizer': 'f52677617f729c21c7440fd52d69b7e570313836a9deb21519425b806043a311',
    'runner': '2f00052c1088f8d13f1b106f25981dc95ff081b2a23841413cca3a7896e0b262',
    'ppo': 'ec0286fdbdd30360b21f2bea314f676058793f9241649d30545d656a961a22a2',
}
PAYLOAD_SCHEMA = 'textop.mixed_ppo1000_actor_RMS.v1'


def _sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _require_environment():
    if not __debug__:
        raise RuntimeError('optimized Python is not allowed')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('the future CPU supervisor must disable CUDA')


def audit_and_payload(state, training_binding):
    """Inspect a supplied CPU state and build a native actor/RMS payload.

    The caller must separately verify the file bytes, content digest, complete
    run evidence, source provenance, exclusivity, ownership and resources.
    This function executes no optimizer step and does not apply CUDA RNG.
    """
    _require_environment()
    import numpy as np
    import torch
    from rsl_rl.modules.actor_critic import ActorCritic
    from rsl_rl.modules.normalizer import EmpiricalNormalization
    from augmentation2526_mixed_ppo_runtime_v1 import clone, content, equal

    assert not torch.cuda.is_initialized() and torch.cuda.device_count() == 0
    torch.set_num_threads(4)
    for name, cls in [('actor', ActorCritic), ('normalizer', EmpiricalNormalization)]:
        assert _sha(inspect.getfile(cls)) == NATIVE_SHA[name]
    native = pathlib.Path(inspect.getfile(ActorCritic)).parent.parent
    assert _sha(native/'runners/on_policy_runner.py') == NATIVE_SHA['runner']
    assert _sha(native/'algorithms/ppo.py') == NATIVE_SHA['ppo']
    meta = validate_mixed_fullstate_metadata(state, training_binding)
    tensors = 0

    def finite_cpu(v):
        nonlocal tensors
        if torch.is_tensor(v):
            assert v.device.type == 'cpu' and bool(torch.isfinite(v).all())
            tensors += 1
        elif isinstance(v, np.ndarray):
            assert np.isfinite(v).all()
        elif isinstance(v, (float, np.floating)):
            assert math.isfinite(float(v))
        elif isinstance(v, dict):
            for child in v.values(): finite_cpu(child)
        elif isinstance(v, (tuple, list)):
            for child in v: finite_cpu(child)
    finite_cpu(state)
    original = content(state)
    rng = state['rng']
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])
    for key in ('cpu', 'cuda_device1'):
        assert rng[key].dtype == torch.uint8 and rng[key].ndim == 1 and rng[key].numel() > 0
    torch.Generator(device='cpu').set_state(rng['cpu'])
    assert ('std' in state['model']) != ('log_std' in state['model'])
    noise = 'scalar' if 'std' in state['model'] else 'log'
    model = ActorCritic(431, 557, 29, actor_hidden_dims=[2048, 1024, 512],
        critic_hidden_dims=[2048, 1024, 512], activation='elu', noise_std_type=noise).cpu()
    model.load_state_dict(state['model'], strict=True)
    assert equal(model.state_dict(), state['model'])
    params = list(model.parameters())
    opt = state['optimizer']
    ids = [i for group in opt['param_groups'] for i in group['params']]
    assert len(ids) == len(set(ids)) == len(params) == len(opt['state'])
    for key, param in zip(ids, params):
        item = opt['state'][key]
        assert set(item) == {'step', 'exp_avg', 'exp_avg_sq'}
        assert item['step'].numel() == 1 and float(item['step']) == 20000
        assert item['exp_avg'].shape == item['exp_avg_sq'].shape == param.shape
        assert item['exp_avg'].dtype == item['exp_avg_sq'].dtype == param.dtype
    for group in opt['param_groups']:
        assert group['lr'] == state['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=float(state['learning_rate']))
    optimizer.load_state_dict(clone(opt))
    assert equal(optimizer.state_dict(), opt) and equal(model.state_dict(), state['model'])
    normalizers = []
    for name, dim in [('obs_normalizer', 431), ('critic_normalizer', 557)]:
        values = state[name]
        assert values['count'].dtype == torch.long and values['count'].numel() == 1 and int(values['count']) > 0
        for key in ('_mean', '_std', '_var'):
            assert values[key].shape == (1, dim) and values[key].dtype == torch.float32
        assert bool((values['_std'] >= 0).all()) and bool((values['_var'] >= 0).all())
        norm = EmpiricalNormalization([dim], eps=1e-2, until=1e8)
        norm.load_state_dict(values, strict=True)
        assert equal(norm.state_dict(), values)
        norm.eval().requires_grad_(False); normalizers.append(norm)
    model.eval().requires_grad_(False)
    norm = normalizers[0]
    probes = [torch.zeros((1, 431), dtype=torch.float32),
        torch.linspace(-2, 2, 7*431, dtype=torch.float32).reshape(7, 431)]
    before = content(dict(model=model.state_dict(), normalizer=norm.state_dict()))
    cpu_rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        expected = [model.act_inference(norm(x)).clone() for x in probes]
    assert all(x.shape == (len(p), 29) and bool(torch.isfinite(x).all()) for x, p in zip(expected, probes))
    assert any(bool((x != 0).any()) for x in expected)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert content(dict(model=model.state_dict(), normalizer=norm.state_dict())) == before
    payload = dict(schema=PAYLOAD_SCHEMA, training_binding=training_binding,
        actor=clone(model.actor.state_dict()), obs_normalizer=clone(norm.state_dict()),
        actor_observation_dim=431, action_dim=29, hidden=[2048, 1024, 512], activation='elu',
        normalization=dict(kind='EmpiricalNormalization', eps=.01, until=1e8, eval_mode=True),
        input='raw assembled431D policy observation before RMS; no critic terms',
        output='raw deterministic29D policy action before environment actuator scaling',
        policy_hz=50, future_reference_frames=5, action_clamp_added=False,
        native_sources=NATIVE_SHA, heldout_tracking_accepted=False, online_latency_accepted=False,
        physics_trajectory_replay_claimed=False)
    assert content(state) == original and not torch.cuda.is_initialized()
    return payload, dict(fullstate_metadata=meta, finite_cpu_tensors=tensors,
        full_model_and_cpu_Adam_restored_exact=True, optimizer_updates_performed=0,
        CUDA_RNG_restored=False, synthetic_probe_only=True,
        fullstate_content_sha256=original, payload_content_sha256=content(payload),
        probe_output_hashes=[content(x) for x in expected]), expected, probes


def independent_reload(payload, expected, probes):
    """Reconstruct actor and RMS; caller must run independent process separately."""
    _require_environment()
    import torch
    from rsl_rl.modules.normalizer import EmpiricalNormalization
    from augmentation2526_mixed_ppo_runtime_v1 import content, equal
    assert not torch.cuda.is_initialized() and torch.cuda.device_count() == 0
    assert payload['schema'] == PAYLOAD_SCHEMA and payload['native_sources'] == NATIVE_SHA
    assert (payload['actor_observation_dim'], payload['action_dim'], payload['policy_hz'],
        payload['future_reference_frames']) == (431, 29, 50, 5)
    assert payload['hidden'] == [2048, 1024, 512] and payload['activation'] == 'elu'
    assert payload['action_clamp_added'] is False
    assert payload['normalization'] == dict(kind='EmpiricalNormalization', eps=.01, until=1e8, eval_mode=True)
    assert _sha(inspect.getfile(EmpiricalNormalization)) == NATIVE_SHA['normalizer']
    sizes = [431, 2048, 1024, 512, 29]; layers = []
    for i, (left, right) in enumerate(zip(sizes, sizes[1:])):
        layers.append(torch.nn.Linear(left, right))
        if i < 3: layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers)
    actor.load_state_dict(payload['actor'], strict=True)
    norm = EmpiricalNormalization([431], eps=.01, until=1e8)
    norm.load_state_dict(payload['obs_normalizer'], strict=True)
    actor.eval().requires_grad_(False); norm.eval().requires_grad_(False)
    before = content(dict(actor=actor.state_dict(), norm=norm.state_dict()))
    cpu_rng = torch.get_rng_state().clone()
    with torch.inference_mode(): actual = [actor(norm(x)).clone() for x in probes]
    assert equal(actual, expected) and torch.equal(cpu_rng, torch.get_rng_state())
    assert content(dict(actor=actor.state_dict(), norm=norm.state_dict())) == before
    assert not torch.cuda.is_initialized()
    return dict(actor_RMS_reload_exact=True, synthetic_probe_only=True,
        independent_process_required=True, heldout_or_online_quality=False)

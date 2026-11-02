"""Draft construction of the mixed PPO5000 actor and learned normalizer.

This is not an executable entry point or a provenance/lock gate. A future
supervisor must first accept the natural training, export and independent
verify closures, verify the payload bytes and hold the common lock. Importing
this module does not load torch, a model, a checkpoint or the simulator.
"""

NATIVE = {
    'actor': '51ce8c6504a7434f86e5188ccb11e9fc981e46f764c00e32be3e8febe7112ae2',
    'normalizer': 'f52677617f729c21c7440fd52d69b7e570313836a9deb21519425b806043a311',
    'runner': '2f00052c1088f8d13f1b106f25981dc95ff081b2a23841413cca3a7896e0b262',
    'ppo': 'ec0286fdbdd30360b21f2bea314f676058793f9241649d30545d656a961a22a2',
}


def validate_metadata(payload, expected_training_binding):
    """Validate the exporter contract; this alone does not authenticate it."""
    if not isinstance(payload, dict) or not isinstance(expected_training_binding, dict) or not expected_training_binding:
        raise ValueError('missing actor payload or training binding')
    if payload.get('training_binding') != expected_training_binding:
        raise ValueError('actor belongs to a different training run')
    checks = {
        'schema': 'textop.mixed_ppo5000_actor_RMS.v1',
        'actor_observation_dim': 431, 'action_dim': 29, 'policy_hz': 50,
        'future_reference_frames': 5, 'hidden': [2048, 1024, 512],
        'activation': 'elu', 'native_sources': NATIVE,
        'normalization': dict(kind='EmpiricalNormalization', eps=.01,
                              until=1e8, eval_mode=True),
    }
    for key, value in checks.items():
        if payload.get(key) != value:
            raise ValueError('incompatible actor payload: ' + key)
    if payload.get('action_clamp_added') is not False:
        raise ValueError('unexpected action clamp')
    if not isinstance(payload.get('actor'), dict) or not isinstance(payload.get('obs_normalizer'), dict):
        raise ValueError('missing actor or normalizer state')


def build_actor(payload, expected_training_binding, device):
    """Build the frozen native actor; invoke only inside an authorized job."""
    validate_metadata(payload, expected_training_binding)
    if device not in ('cpu', 'cuda:1'):
        raise ValueError('unexpected inference device')
    import hashlib
    import inspect
    import torch
    from rsl_rl.modules.normalizer import EmpiricalNormalization
    with open(inspect.getfile(EmpiricalNormalization), 'rb') as source:
        if hashlib.sha256(source.read()).hexdigest() != NATIVE['normalizer']:
            raise ValueError('native normalizer source mismatch')
    sizes = [431, 2048, 1024, 512, 29]
    layers = []
    for index, (left, right) in enumerate(zip(sizes, sizes[1:])):
        layers.append(torch.nn.Linear(left, right))
        if index < 3:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers)
    normalizer = EmpiricalNormalization([431], eps=.01, until=1e8)
    actor.load_state_dict(payload['actor'], strict=True)
    normalizer.load_state_dict(payload['obs_normalizer'], strict=True)
    actor.to(device).eval().requires_grad_(False)
    normalizer.to(device).eval().requires_grad_(False)
    for value in list(actor.state_dict().values()) + list(normalizer.state_dict().values()):
        if not torch.isfinite(value).all().item():
            raise ValueError('nonfinite actor or normalizer state')

    def infer(observations):
        if observations.ndim != 2 or observations.shape[1] != 431 or observations.device != torch.device(device):
            raise ValueError('unexpected policy observations')
        if not torch.isfinite(observations).all().item():
            raise ValueError('nonfinite policy observations')
        with torch.inference_mode():
            actions = actor(normalizer(observations))
        if actions.shape != (observations.shape[0], 29) or not torch.isfinite(actions).all().item():
            raise ValueError('invalid policy action')
        return actions
    return infer

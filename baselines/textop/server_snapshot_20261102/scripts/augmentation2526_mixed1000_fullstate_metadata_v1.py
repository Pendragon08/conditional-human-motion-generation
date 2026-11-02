"""Local draft: validate mixed PPO fullstate metadata, without loading a model.

This is not a training acceptance, a checkpoint loader, or an authorization to
run an audit. A future trusted supervisor must independently authenticate the
training binding, own the shared lock, and validate all tensor contents.
"""
import math

SCHEMA = 'textop.stage4_mixed_train_fullstate.v1'
RUN = 'user_augmentation2526_mixed_ppo_train_v1'
REQUEST_SHA = 'b3652704ef2ce6bd4c5b2f433de37d17329b4a57b1ec1d1891fdd880b0ad04f9'
ADS_NAMES = frozenset(('failed_motion_count', 'success_motion_count',
    'current_failed_motion_count', 'current_success_motion_count',
    'failed_total_count', 'success_total_count', 'raw_failed', 'raw_success'))


def validate_mixed_fullstate_metadata(state, binding):
    """Validate structural metadata only; tensor contents need separate audit."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(isinstance(state, dict) and isinstance(binding, dict), 'expected dictionaries')
    require(binding.get('kind') == 'MIXED_PPO1000_FULLSTATE_CONTENT_UNAUDITED' and
            binding.get('run') == RUN and binding.get('request_sha256') == REQUEST_SHA and
            binding.get('fullstate_content_independently_audited') is False,
            'wrong training binding')
    source = binding.get('input_library_binding')
    require(isinstance(source, dict) and set(source) == {'original', 'generated_audit'} and
            all(isinstance(source[k], dict) and source[k] for k in source), 'missing mixed libraries')
    expected = {'schema', 'completed_iterations', 'optimizer_updates', 'input_binding',
        'model', 'optimizer', 'obs_normalizer', 'critic_normalizer', 'learning_rate',
        'rng', 'adaptive_sampling', 'runner_clock', 'physics_continuation', 'algorithm'}
    require(set(state) == expected and state['schema'] == SCHEMA, 'wrong fullstate schema')
    require(type(state['completed_iterations']) is int and state['completed_iterations'] == 1000 and
            type(state['optimizer_updates']) is int and state['optimizer_updates'] == 20000,
            'wrong training boundary')
    require(state['input_binding'] == source, 'mixed library binding differs')
    require(state['physics_continuation'] == 'new episodes; simulator caches/in-flight episodes not serialized' and
            state['algorithm'] == 'fresh one-stage PPO; no teacher/DAgger/distillation', 'unexpected continuation')
    require(isinstance(state['rng'], dict) and set(state['rng']) == {'python', 'numpy', 'cpu', 'cuda_device1'}, 'wrong RNG structure')
    require(isinstance(state['adaptive_sampling'], dict) and set(state['adaptive_sampling']) == ADS_NAMES, 'wrong sampling structure')
    require(isinstance(state['runner_clock'], dict) and set(state['runner_clock']) == {'tot_time', 'tot_timesteps'}, 'wrong clock structure')
    clock = state['runner_clock']
    require(type(clock['tot_time']) in (int, float) and math.isfinite(clock['tot_time']) and clock['tot_time'] > 0 and
            type(clock['tot_timesteps']) is int and clock['tot_timesteps'] == 1000*24*8192, 'wrong clock values')
    rate = state['learning_rate']
    require(type(rate) in (int, float) and math.isfinite(rate) and rate > 0, 'wrong learning rate')
    normalizer_keys = {'_mean', '_var', '_std', 'count'}
    for key in ('obs_normalizer', 'critic_normalizer'):
        require(isinstance(state[key], dict) and set(state[key]) == normalizer_keys, 'wrong normalizer structure')
    require(isinstance(state['model'], dict) and bool(state['model']), 'empty model')
    opt = state['optimizer']
    require(isinstance(opt, dict) and set(opt) == {'state', 'param_groups'} and
            isinstance(opt['state'], dict) and bool(opt['state']) and
            isinstance(opt['param_groups'], list) and bool(opt['param_groups']), 'empty optimizer')
    return {'metadata_only': True, 'mixed_schema': SCHEMA, 'completed_iterations': 1000,
            'optimizer_updates': 20000, 'tensor_contents_audited': False,
            'model_or_optimizer_reloaded': False, 'physics_trajectory_replay_claimed': False}

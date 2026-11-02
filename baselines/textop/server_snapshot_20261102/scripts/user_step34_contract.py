"""One-stage PPO policy contract, deliberately separate from the557D teacher."""
BASE_TERMS=[('reference_joint_pos_vel',290),('reference_pelvis_position',15),
    ('reference_pelvis_orientation_6D',30),('projected_gravity',3),('base_lin_ang_velocity',6),
    ('joint_pos_vel',58),('last_action',29)]
PRIVILEGED_TERMS=[('key_body_position',42),('key_body_orientation_6D',84)]


def validate(policy):
    assert policy['method']=='one_stage_PPO' and policy['dagger_or_distillation'] is False
    assert policy['actor_terms']==[list(x) for x in BASE_TERMS]
    assert policy['critic_terms']==[list(x) for x in BASE_TERMS+PRIVILEGED_TERMS]
    assert sum(x[1] for x in policy['actor_terms'])==policy['actor_observation_dim']==431
    assert sum(x[1] for x in policy['critic_terms'])==policy['critic_observation_dim']==557
    assert policy['future_reference_frames']==5 and policy['dof']==policy['action_dim']==29
    assert policy['actor_hidden']==policy['critic_hidden']==[2048,1024,512]
    assert policy['activation']=='elu' and policy['simulation_hz']==200 and policy['policy_hz']==50
    assert policy['rollout_steps_per_env']==24 and policy['num_envs']==8192
    assert policy['learning_rate']==.001 and policy['minibatches']==4 and policy['ppo_epochs']==5 and policy['clip_param']==.2
    assert policy['fresh_actor_critic_and_Adam_initialization'] is True
    assert policy['old_privileged_teacher_training_resumed'] is False
    assert type(policy['training_authorized']) is bool
    if policy['training_authorized']:
        assert isinstance(policy['accepted_mixed_dataset'],dict) and policy['accepted_mixed_dataset']
        assert policy['real_RL_witnesses_accepted'] is True
    else:
        assert policy['real_RL_witnesses_accepted'] is False
    assert policy['new_RL_training_updates']==0
    assert policy['pfail_conditional_regularization'] is False
    return policy


def parameter_count():
    def network(dims):return sum((a+1)*b for a,b in zip(dims[:-1],dims[1:]))
    actor=network([431,2048,1024,512,29]);critic=network([557,2048,1024,512,1])
    total=actor+critic+29
    return dict(actor_parameters=actor,critic_parameters=critic,gaussian_std_parameters=29,total_parameters=total,
                float32_model_and_two_Adam_moments_payload_bytes=total*12,
                excludes_optimizer_metadata_normalizers_RNG_sampler_and_simulator_state=True)

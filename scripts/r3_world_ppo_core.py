"""Causal predictive features for real-rollout PPO; no imagined rewards or IK."""
from __future__ import annotations
import hashlib
import json
import torch
from torch import nn
from rsl_rl.modules import ActorCritic

HISTORY = 10
INPUT = 92  # original observable proprioception 63 + previous action 29
LATENT = 128
HORIZONS = (1, 5, 10)
CONTINUOUS = 48
TARGET = 50
SCHEMA = {
    'version': 1, 'history': HISTORY, 'input': INPUT, 'latent': LATENT,
    'input_source': 'original noisy policy observation[0:92], no privileged state',
    'input_order': ['root_linear_velocity_b:3', 'root_angular_velocity_b:3',
                    'projected_gravity_b:3', 'relative_joint_position:27',
                    'relative_joint_velocity:27', 'previous_action:29'],
    'targets': ['absolute_motion_joint_position:27', 'projected_gravity_b:3',
                'root_linear_velocity_b:3', 'root_angular_velocity_b:3',
                'two_feet_relative_root_position_b:6', 'two_feet_velocity_xy_w:4',
                'root_height:1', 'root_vertical_velocity_w:1', 'two_ground_contact:2'],
    'target_continuous_delta': True, 'contact_force_threshold_N': 5.0,
    'future_action_in_actor': False, 'step_dt_s': 0.02,
}
# The 626D policy has 27 tracking joints plus 29 actions, not 29 tracked joints.
assert sum([3,3,3,27,27,29]) == INPUT
SCHEMA_HASH = hashlib.sha256(json.dumps(SCHEMA, sort_keys=True,
                                        separators=(',', ':')).encode()).hexdigest()


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(INPUT, LATENT, batch_first=True)
        self.latent_norm = nn.LayerNorm(LATENT, elementwise_affine=False)
        self.heads = nn.ModuleDict({str(k): nn.Sequential(
            nn.Linear(LATENT + 29*k, 256), nn.ELU(), nn.Linear(256, TARGET)
        ) for k in HORIZONS})
        self.register_buffer('input_mean', torch.zeros(INPUT))
        self.register_buffer('input_std', torch.ones(INPUT))
        self.register_buffer('delta_mean', torch.zeros(CONTINUOUS))
        self.register_buffer('delta_std', torch.ones(CONTINUOUS))

    def encode(self, history):
        valid=history.abs().sum(-1,keepdim=True)>0
        normalized=torch.where(valid,(history-self.input_mean)/self.input_std,0.)
        return self.latent_norm(self.encoder(normalized)[1][0])

    def predict(self, history, future_actions, state_only=False):
        latent = self.encode(history)
        return {k: self.heads[str(k)](torch.cat([
            latent, torch.zeros_like(future_actions[:, :k]).flatten(1) if state_only
            else future_actions[:, :k].flatten(1)
        ], -1)) for k in HORIZONS}


class HistoryBuffer:
    """Finite ordered windows; rows after an environment reset are zero padded."""
    def __init__(self, n, device):
        self.value = torch.zeros(n, HISTORY, INPUT, device=device)
        self.reset_mask = torch.ones(n, dtype=torch.bool, device=device)

    def append(self, observable, done=None):
        self.value[:, :-1] = self.value[:, 1:].clone()
        if done is not None:
            self.value[done.bool()] = 0
            self.reset_mask = done.bool().clone()
        else:
            self.reset_mask.zero_()
        self.value[:, -1] = observable[:, :INPUT]
        return self.value


class AugmentedActorCritic(ActorCritic):
    """Uses unchanged RSL PPO distribution/value APIs and zero-init new input.

    W latent is already cached in rollout observations. History control stores
    the entire ordered raw window and recomputes a differentiable encoder, never
    reconstructing a GRU from unordered minibatch samples.
    """
    def __init__(self, arm, original_state, encoder_state=None):
        self.arm = arm
        super().__init__(626 if arm == 'P0' else 754,
                         626 if arm == 'P0' else 754, 29,
                         actor_hidden_dims=[512,256,256],
                         critic_hidden_dims=[512,256,256], activation='elu')
        adapted = {}
        for key, value in original_state.items():
            if key in ['actor.0.weight', 'critic.0.weight'] and arm != 'P0':
                extended = torch.zeros(value.shape[0], 754, dtype=value.dtype)
                extended[:, :626] = value
                adapted[key] = extended
            else:
                adapted[key] = value
        self.load_state_dict(adapted)
        self.register_buffer('input_mean',torch.zeros(INPUT))
        self.register_buffer('input_std',torch.ones(INPUT))
        self.latent_norm=nn.LayerNorm(LATENT,elementwise_affine=False)
        if arm in ['P-history','P-cap']:
            self.history_encoder = nn.GRU(INPUT, LATENT, batch_first=True)
            if encoder_state:
                self.history_encoder.load_state_dict(encoder_state)

    def features(self, observations):
        if self.arm == 'P-history':
            history = observations[:, 626:].reshape(-1,HISTORY,INPUT)
            valid=history.abs().sum(-1,keepdim=True)>0
            normalized=torch.where(valid,(history-self.input_mean)/self.input_std,0.)
            latent=self.latent_norm(self.history_encoder(normalized)[1][0])
            return torch.cat([observations[:, :626], latent], -1)
        if self.arm == 'P-cap':
            current=observations[:,626:626+INPUT]
            current=(current-self.input_mean)/self.input_std
            current=current.clone(); current[:,-29:]=0
            latent=self.latent_norm(self.history_encoder(current[:,None,:].expand(-1,HISTORY,-1))[1][0])
            return torch.cat([observations[:, :626],latent], -1)
        return observations

    def update_distribution(self, observations):
        return super().update_distribution(self.features(observations))

    def act_inference(self, observations):
        return self.actor(self.features(observations))

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(self.features(critic_observations))


def prediction_loss(model, history, actions, targets, state_only=False):
    predictions = model.predict(history, actions, state_only)
    loss = 0
    for k in HORIZONS:
        continuous = (targets[k][:, :CONTINUOUS]-model.delta_mean)/model.delta_std
        loss = loss + (predictions[k][:,:CONTINUOUS]-continuous).square().mean()
        loss = loss + 0.1*nn.functional.binary_cross_entropy_with_logits(
            predictions[k][:,CONTINUOUS:], targets[k][:,CONTINUOUS:])
    return loss/len(HORIZONS), predictions


def decoded(model, prediction):
    return torch.cat([prediction[:,:CONTINUOUS]*model.delta_std+model.delta_mean,
                      prediction[:,CONTINUOUS:].sigmoid()], -1)

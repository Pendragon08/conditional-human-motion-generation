"""SuperMimic-style world-model/BPTT adaptation for the existing Pick PhysX task.

The native TextOp source supplies unchanged MLP/normalization primitives. The
state, live policy and learned reward/termination heads are project adaptations
for a 29-body-joint robot with 14 finger joints and a free object. No PPO loss.
"""
import ast
import hashlib
from pathlib import Path
import types

import torch
from torch import nn
from torch.nn import functional as F

from stage3_pick29_core import actor_observation, quat_apply, quat_conjugate, quat_mul

NATIVE_NETWORK_SHA = '72bdb2bf11f927663dec82ef00f75123dfe810cba9879fee455faac4042b1efe'
STATE_DIM, ACTION_DIM, REF_DIM = 112, 31, 109


def native_primitives(source_root):
    path = Path(source_root)/'supermimic/networks.py'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != NATIVE_NETWORK_SHA:
        raise ValueError('pinned SuperMimic network source changed')
    names = {'mlp', 'RunningNorm', 'run_mlp'}
    nodes = [n for n in ast.parse(raw).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError('native primitive definitions missing')
    module = types.ModuleType('pinned_supermimic_primitives')
    module.__dict__.update(torch=torch, nn=nn, AMP_DTYPE=None)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module


def normalize_quat(q):
    return q/q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_log(q):
    q = normalize_quat(q)
    q = torch.where(q[..., :1] < 0, -q, q)
    length = q[..., 1:].norm(dim=-1, keepdim=True)
    angle = 2*torch.atan2(length, q[..., :1].clamp_min(1e-8))
    scale = torch.where(length > 1e-5, angle/length.clamp_min(1e-8), 2.+length.square()/3.)
    return q[..., 1:]*scale


def quat_exp(v):
    angle = v.norm(dim=-1, keepdim=True)
    scale = .5*torch.sinc(angle/(2*torch.pi))
    return normalize_quat(torch.cat((torch.cos(angle/2), v*scale), -1))


def delta_target(s, s1):
    return torch.cat((s1[..., :3]-s[..., :3],
        quat_log(quat_mul(quat_conjugate(s[..., 3:7]), s1[..., 3:7])),
        s1[..., 7:102]-s[..., 7:102],
        quat_log(quat_mul(quat_conjugate(s[..., 102:106]), s1[..., 102:106])),
        s1[..., 106:112]-s[..., 106:112]), -1)


def integrate(s, delta):
    # 112 flat state -> 110 increments: two quaternion updates use rotation vectors.
    return torch.cat((s[..., :3]+delta[..., :3],
        normalize_quat(quat_mul(s[..., 3:7], quat_exp(delta[..., 3:6]))),
        s[..., 7:102]+delta[..., 6:101],
        normalize_quat(quat_mul(s[..., 102:106], quat_exp(delta[..., 101:104]))),
        s[..., 106:112]+delta[..., 104:110]), -1)


def live_obs(s, a_prev, default_q, default_qd):
    root_q = s[..., 3:7]
    inverse = quat_conjugate(root_q)
    gravity = torch.zeros_like(s[..., :3]); gravity[..., 2] = -1
    return actor_observation(quat_apply(inverse, s[..., 10:13]), quat_apply(inverse, gravity),
        s[..., 13:42]-default_q[:29], s[..., 56:85]-default_qd[:29], a_prev,
        s[..., :3], root_q, s[..., 99:102], s[..., 102:106])


def hard_hands(action, straight_through=False):
    hard = torch.where(action[..., -2:] < 0, -torch.ones_like(action[..., -2:]),
                       torch.ones_like(action[..., -2:]))
    hands = action[..., -2:]+(hard-action[..., -2:]).detach() if straight_through else hard
    return torch.cat((action[..., :29], hands), -1)


def private_reference(ref):
    return torch.cat((ref['q'], ref['root_pos'], ref['root_quat'],
        ref['keypoints'].flatten(-2), ref['hand_closed'], ref['right_wrist_quat'],
        ref['after_grasp'].to(ref['q'].dtype).unsqueeze(-1)), -1)


class LivePolicy(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.norm = native.RunningNorm(102)
        self.net = native.mlp(102, [512, 256, 256], 31)
        with torch.no_grad():
            self.net[-1].weight.mul_(.01); self.net[-1].bias.zero_()

    def forward(self, observation):
        # Sole policy argument is the existing102 live observation.
        return self.net(self.norm(observation))


class LiveCritic(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.norm = native.RunningNorm(102)
        self.net = native.mlp(102, [512, 256, 256], 1)

    def forward(self, observation):
        return self.net(self.norm(observation)).squeeze(-1)


class PickWorldModel(nn.Module):
    def __init__(self, native, hidden=(2048, 2048, 2048)):
        super().__init__()
        self.input_norm = native.RunningNorm(STATE_DIM+ACTION_DIM)
        self.delta_norm = native.RunningNorm(110, eps=1e-4)
        self.net = native.mlp(STATE_DIM+ACTION_DIM, list(hidden), 110,
                              act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.head_norm = native.RunningNorm(STATE_DIM+2*ACTION_DIM+REF_DIM)
        self.reward_norm = native.RunningNorm(1)
        self.head = native.mlp(STATE_DIM+2*ACTION_DIM+REF_DIM, [256, 256], 2)

    def step(self, s, action):
        x = self.input_norm(torch.cat((s, action), -1))
        delta = self.net(x)*self.delta_norm.std()+self.delta_norm.mean
        return integrate(s, delta)

    def reward_termination(self, next_state, action, previous_action, reference):
        x = torch.cat((next_state, action, previous_action, reference), -1)
        output = self.head(self.head_norm(x))
        reward = output[..., 0]*self.reward_norm.std()[0]+self.reward_norm.mean[0]
        return reward, output[..., 1]

    @torch.no_grad()
    def update_norms(self, data):
        self.input_norm.update(torch.cat((data['s'], data['a']), -1))
        self.delta_norm.update(delta_target(data['s'], data['s1']))
        self.head_norm.update(torch.cat((data['s1'], data['a'], data['a_prev'], data['ref']), -1))
        self.reward_norm.update(data['r'].unsqueeze(-1))


def fit_world_model(wm, optimizer, batch):
    """Autoregressive k-step fit; mask excludes transitions after an episode ends."""
    state = batch['s'][:, 0]
    alive = torch.ones(state.shape[0], device=state.device)
    dynamics, rewards, terminals, denominator = [], [], [], []
    for t in range(batch['s'].shape[1]):
        predicted = wm.step(state, batch['a'][:, t])
        error = (delta_target(predicted, batch['s1'][:, t])/wm.delta_norm.std()).square().mean(-1)
        reward, logit = wm.reward_termination(batch['s1'][:, t], batch['a'][:, t],
            batch['a_prev'][:, t], batch['ref'][:, t])
        re = ((reward-batch['r'][:, t])/wm.reward_norm.std()[0]).square()
        te = F.binary_cross_entropy_with_logits(logit, batch['term'][:, t].float(), reduction='none')
        dynamics.append((error*alive).sum()); rewards.append((re*alive).sum())
        terminals.append((te*alive).sum()); denominator.append(alive.sum())
        alive = alive*(1-batch['done'][:, t].float())
        state = predicted
    norm = torch.stack(denominator).sum().clamp_min(1)
    dyn = torch.stack(dynamics).sum()/norm
    rew = torch.stack(rewards).sum()/norm
    term = torch.stack(terminals).sum()/norm
    loss = dyn+rew+term
    if not torch.isfinite(loss): raise RuntimeError('nonfinite world-model loss')
    optimizer.zero_grad(set_to_none=True); loss.backward()
    grad = nn.utils.clip_grad_norm_(wm.parameters(), 1.)
    if not torch.isfinite(grad): raise RuntimeError('nonfinite world-model gradient')
    optimizer.step()
    return {'world_model_loss': float(loss.detach()), 'dynamics_loss': float(dyn.detach()),
            'reward_loss': float(rew.detach()), 'termination_loss': float(term.detach())}


def improve_policy(policy, critic, wm, policy_optimizer, critic_optimizer, starts,
                   bank, default_q, default_qd, horizon=8, gamma=.99, lam=.95,
                   critic_epochs=5, critic_minibatches=4):
    """Actual BPTT through a frozen world model, then imagined GAE critic fitting."""
    for module in [wm, critic]:
        for parameter in module.parameters(): parameter.requires_grad_(False)
    state, previous = starts['s'], starts['a_prev']
    observations, rs, ps = [live_obs(state, previous, default_q, default_qd)], [], []
    try:
        for t in range(horizon):
            action = hard_hands(policy(observations[-1]), straight_through=True)
            next_state = wm.step(state, action)
            seconds = (starts['step'].float()+t+1)*.02
            ref = private_reference(bank.query(starts['id'].long(), seconds))
            reward, logit = wm.reward_termination(next_state, action, previous, ref)
            rs.append(reward); ps.append(torch.sigmoid(logit))
            observations.append(live_obs(next_state, action, default_q, default_qd))
            state, previous = next_state, action
        rewards, terminals = torch.stack(rs), torch.stack(ps)
        survival = torch.cumprod(1-terminals, 0)
        before = torch.cat((torch.ones_like(survival[:1]), survival[:-1]), 0)
        discount = gamma**torch.arange(horizon, device=state.device, dtype=state.dtype)
        objective = (discount[:, None]*before*rewards).sum(0)
        objective = objective+gamma**horizon*survival[-1]*critic(observations[-1])
        loss = -objective.mean()
        if not torch.isfinite(loss): raise RuntimeError('nonfinite imagined objective')
        policy_optimizer.zero_grad(set_to_none=True); loss.backward()
        gradient = nn.utils.clip_grad_norm_(policy.parameters(), 1.)
        nn.utils.clip_grad_value_(policy.parameters(), .1)
        if not torch.isfinite(gradient): raise RuntimeError('nonfinite policy gradient')
        policy_optimizer.step()
    finally:
        for module in [wm, critic]:
            for parameter in module.parameters(): parameter.requires_grad_(True)
    x = torch.stack(observations).detach()
    critic.norm.update(x)
    with torch.no_grad():
        values = critic(x)
        ret = values[-1]
        targets = []
        for t in range(horizon-1, -1, -1):
            ret = rewards[t].detach()+gamma*(1-terminals[t].detach())*((1-lam)*values[t+1]+lam*ret)
            targets.append(ret)
        target = torch.stack(targets[::-1]).flatten()
    xf = x[:-1].flatten(0, 1)
    value_loss = torch.zeros((), device=state.device)
    for _ in range(critic_epochs):
        for indices in torch.tensor_split(torch.randperm(len(xf), device=state.device), critic_minibatches):
            if len(indices)==0: continue
            value_loss = (critic(xf[indices])-target[indices]).square().mean()
            if not torch.isfinite(value_loss): raise RuntimeError('nonfinite critic objective')
            critic_optimizer.zero_grad(set_to_none=True); value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.); critic_optimizer.step()
    return {'imagined_policy_loss': float(loss.detach()), 'critic_loss': float(value_loss.detach()),
            'policy_gradient_norm': float(gradient.detach()),
            'imagined_reward_mean': float(rewards.detach().mean()),
            'imagined_survival_mean': float(survival[-1].detach().mean())}

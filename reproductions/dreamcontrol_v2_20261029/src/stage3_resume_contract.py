"""RSL-RL checkpoints store a completed zero-based iteration, not the next index."""
import torch


def resume_counts(saved_iteration_index, total_updates):
    completed=int(saved_iteration_index)+1
    if not 0<completed<total_updates:
        raise ValueError('resume checkpoint must precede the requested total budget')
    return completed,total_updates-completed


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def verify_restoration(runner,saved):
    checks={
        'model':equal_tree(runner.alg.policy.state_dict(),saved['model_state_dict']),
        'adam':equal_tree(runner.alg.optimizer.state_dict(),saved['optimizer_state_dict']),
        'actor_normalizer':equal_tree(runner.obs_normalizer.state_dict(),saved['obs_norm_state_dict']),
        'critic_normalizer':equal_tree(runner.privileged_obs_normalizer.state_dict(),saved['privileged_obs_norm_state_dict'])}
    if not all(checks.values()):raise RuntimeError('resume state mismatch: '+repr(checks))
    return checks

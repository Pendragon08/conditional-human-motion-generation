"""Numerical checks of Punch preparation; no simulator or policy updates."""
import inspect
import json
import torch

from stage3_punch29_core import (punch_goal_error_squared, punch_goal_reward,
                                 punch_live_observation, sample_punch_conditions)
from stage3_pick29_core import quat_apply, quat_mul


def main():
    torch.set_num_threads(2)
    passed = []
    # The legacy norm<0.05 predicate would reject the valid 0.10m case.
    points = torch.tensor([[.04, 0., 0.], [.10, 0., 0.], [.23, 0., 0.]], dtype=torch.float64)
    zero = torch.zeros_like(points)
    assert punch_goal_reward(points, zero).tolist() == [1., 1., 0.]
    passed.append('squared_threshold_not_legacy_five_centimetres')
    multi = torch.tensor([[.125, .125, .125], [.125, .125, .25]], dtype=torch.float64)
    assert punch_goal_reward(multi, torch.zeros_like(multi)).tolist() == [1., 0.]
    err = punch_goal_error_squared(points, zero)
    assert torch.equal(punch_goal_reward(points, zero).bool(), err <= .05)
    passed.append('direct_inequality_and_multi_axis_error')

    seed = 20261103
    a = sample_punch_conditions(4096, torch.Generator().manual_seed(seed))
    b = sample_punch_conditions(4096, torch.Generator().manual_seed(seed))
    assert all(torch.equal(a[k], b[k]) for k in a)
    targets = a['right_wrist_target_robot_z_up']
    assert ((targets >= torch.tensor([.45, -.20, .95], dtype=torch.float64))
            & (targets <= torch.tensor([.55, .10, 1.10], dtype=torch.float64))).all()
    assert a['target_frame'].min() == 90 and a['target_frame'].max() == 120
    assert torch.equal(a['target_seconds'], a['target_frame'].double()/20)
    passed.append('seeded_bounds_and_twenty_hz_time')

    n = 3
    q = torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1)
    root = torch.randn(n, 3, generator=torch.Generator().manual_seed(seed))
    target = torch.randn(n, 3, generator=torch.Generator().manual_seed(seed+1))
    args = [torch.zeros(n, 3), torch.tensor([[0., 0., -1.]]).repeat(n, 1),
            torch.zeros(n, 29), torch.zeros(n, 29), torch.zeros(n, 31)]
    obs = punch_live_observation(*args, root, q, target, q)
    assert obs.shape == (n, 102) and torch.isfinite(obs).all()
    rot = torch.tensor([[.7071067811865476, 0., 0., .7071067811865476]]).repeat(n, 1)
    shift = torch.tensor([12., -7., 3.])
    moved = punch_live_observation(*args, quat_apply(rot, root)+shift, quat_mul(rot, q),
                                   quat_apply(rot, target)+shift, quat_mul(rot, q))
    assert torch.allclose(obs, moved, atol=2e-6, rtol=1e-6)
    assert torch.allclose(punch_goal_error_squared(root, target),
                          punch_goal_error_squared(quat_apply(rot, root)+shift,
                                                   quat_apply(rot, target)+shift), atol=1e-5)
    passed.append('live_observation_and_world_frame_invariance')
    assert not any('reference' in k or 'phase' in k or 'time' in k or 'motion_id' in k
                   for k in inspect.signature(punch_live_observation).parameters)
    passed.append('no_reference_phase_or_motion_id_actor_arguments')

    for bad in [torch.zeros(1, 3), torch.full((3, 3), float('nan'))]:
        try:
            punch_goal_reward(root, bad)
        except ValueError:
            continue
        raise AssertionError('invalid target accepted')
    passed.append('shape_and_nonfinite_rejection')
    print(json.dumps({'passed': True, 'checks': passed, 'device': 'cpu',
                      'policy_updates': 0, 'simulator_validation': False,
                      'task_ready_for_training': False}, indent=2))


if __name__ == '__main__':
    main()

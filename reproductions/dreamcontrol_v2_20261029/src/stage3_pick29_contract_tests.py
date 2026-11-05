"""Meaningful CPU tests: frame invariance, quaternion interpolation, and actor/reference separation."""
import inspect
import json
import math
from pathlib import Path
import torch
from stage3_pick29_core import actor_observation,quat_mul,quat_apply,interpolate_quat,ReferenceBank29


def run():
    torch.manual_seed(17)
    n=4
    zero=torch.zeros
    q=torch.tensor([[1.,0,0,0]]).repeat(n,1)
    root=torch.randn(n,3); obj=torch.randn(n,3)
    inputs=[torch.randn(n,3),torch.tensor([[0.,0.,-1.]]).repeat(n,1),torch.randn(n,29),
            torch.randn(n,29),torch.randn(n,31),root,q,obj,q]
    original=actor_observation(*inputs)
    theta=.7
    global_q=torch.tensor([[math.cos(theta/2),0.,0.,math.sin(theta/2)]]).repeat(n,1)
    shift=torch.tensor([2.,-3.,1.])
    transformed=inputs[:5]+[quat_apply(global_q,root)+shift,quat_mul(global_q,q),
                             quat_apply(global_q,obj)+shift,quat_mul(global_q,q)]
    difference=float((actor_observation(*transformed)-original).abs().max())
    assert difference<1e-5
    assert original.shape==(n,102)
    # API structurally excludes reference data, phase, and motion IDs.
    assert set(inspect.signature(actor_observation).parameters)=={
        'angular_velocity','gravity','joint_pos_rel','joint_vel_rel','previous_action',
        'root_pos','root_quat','object_pos','object_quat'}
    endpoints=torch.tensor([[0.,0.,0.,1.]]).repeat(n,1)
    middle=interpolate_quat(q,endpoints,torch.full((n,1),.5))
    rotated=quat_apply(middle,torch.tensor([[1.,0.,0.]]).repeat(n,1))
    assert torch.allclose(rotated,torch.tensor([[0.,1.,0.]]).repeat(n,1),atol=1e-5)
    assert torch.allclose(interpolate_quat(q,-q,torch.full((n,1),.5)),q,atol=1e-6)
    bank=ReferenceBank29.__new__(ReferenceBank29)
    bank.lengths=torch.tensor([2]);bank.tg=torch.tensor([.025])
    bank.data={'q':torch.stack([zero(29),torch.ones(29)])[None],
               'root_quat':torch.stack([q[0],-q[0]])[None],
               'right_wrist_quat':torch.stack([q[0],endpoints[0]])[None],
               'hand_closed':torch.tensor([[[0.,0.],[0.,1.]]])}
    ref=bank.query(torch.tensor([0]),torch.tensor([.025]))
    assert torch.allclose(ref['q'],torch.full((1,29),.5))
    assert torch.equal(ref['hand_closed'],torch.tensor([[0.,0.]]))
    last=bank.query(torch.tensor([0]),torch.tensor([50.]))
    assert torch.equal(last['q'],torch.ones(1,29)) and last['after_grasp'].all()
    invalid=inputs.copy();invalid[2]=zero(n,27)
    try:actor_observation(*invalid)
    except ValueError:pass
    else:raise AssertionError('27DoF input silently accepted')
    return {'tests_passed':7,'actor_obs_dimension':102,'reference_fps':20,
            'world_transform_invariance_max_error':difference,'status':'passed',
            'scope':'CPU data/control contracts; separate real PhysX witness required'}


if __name__=='__main__':
    result=run()
    path=Path(__file__).parents[1]/'reports/core_contract_tests.json'
    with path.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

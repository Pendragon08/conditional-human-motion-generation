"""Seeded CUDA residual interface witness, no simulator or policy training."""
import argparse
import json
from pathlib import Path
import torch
from r3_world_ppo_residual_core import ResidualWorldModel
from r3_world_ppo_core import HORIZONS, decoded

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists()
torch.manual_seed(7)
history=torch.randn(32,10,92,device='cuda:0');actions=torch.randn(32,10,29,device='cuda:0')
coefs={k:torch.randn(921+29*k,50)*.001 for k in HORIZONS}
model=ResidualWorldModel(coefs).to('cuda:0')
pred=model.predict(history,actions)
for k in HORIZONS:
    x=torch.cat([history.flatten(1),actions[:,:k].flatten(1),torch.ones(32,1,device='cuda:0')],-1)
    baseline=x@model._buffers['ridge_'+str(k)]
    assert torch.allclose(decoded(model,pred[k])[:,:48],baseline[:,:48],atol=1e-6)
    assert torch.allclose(decoded(model,pred[k])[:,48:],torch.full((32,2),.5,device='cuda:0'))
zero=model.predict(history,actions,True)
other=model.predict(history,actions*20,True)
assert all(torch.equal(zero[k],other[k]) for k in HORIZONS)
before={k:v.clone() for k,v in model.named_parameters()}
opt=torch.optim.Adam(model.parameters(),lr=.001)
loss=sum(v.square().mean() for v in pred.values());loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
opt.step();assert any(not torch.equal(v,before[k]) for k,v in model.named_parameters())
answer={'status':'COMPLETE','device':torch.cuda.get_device_name(),
        'initial_continuous_matches_ridge':True,'contact_init_probability':.5,
        'future_action_zero_in_both_branches':True,'finite_true_cuda_update':True,
        'diagnostic_aux_updates':1,'formal_PPO_updates':0}
a.output.write_text(json.dumps(answer,indent=2)+'\n')
print('WORLD_RESIDUAL_KERNEL_PASS',json.dumps(answer),flush=True)

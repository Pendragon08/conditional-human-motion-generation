"""Seeded CUDA execution and package-resolution witness; no policy score."""
import hashlib, importlib.metadata as md, importlib.util, json, pathlib, sys
import torch

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
x=torch.randn(128,128,device='cuda:0',requires_grad=True)
y=(x@x.T).square().mean()
y.backward()
torch.cuda.synchronize()
assert torch.isfinite(y) and torch.isfinite(x.grad).all()
root=pathlib.Path(__file__).resolve().parents[1]
package=importlib.util.find_spec('whole_body_tracking').origin
assert str(root/'upstream') in package, package
out={'seed':42,'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'shape':list(x.shape),'loss':y.item(),'gradient_finite':True,'tracking_package':package,'packages':{n:md.version(n) for n in ['rsl-rl-lib','numpy','gymnasium','wandb','hydra-core']}}
(root/'results').mkdir(exist_ok=True)
(root/'results/kernel_witness.json').write_text(json.dumps(out,indent=2)+'\n')
print('PHY_GILE_KERNEL_WITNESS',json.dumps(out))

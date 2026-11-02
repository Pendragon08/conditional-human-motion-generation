"""Small seeded official MVAE CUDA forward/backward; not a training result."""
import json,pathlib,random,sys
import numpy as np
import torch
from robotmdar.model.mld_vae import AutoMldVae
random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.65)
model=AutoMldVae(nfeats=57,latent_dim=[1,128],h_dim=512,ff_size=1024,num_layers=9,num_heads=4).cuda()
history=torch.randn(2,2,57,device='cuda');future=torch.randn(2,8,57,device='cuda')
latent,dist=model.encode(future_motion=future,history_motion=history)
out=model.decode(latent,history,nfuture=8)
loss=(out-future).square().mean();loss.backward()
assert out.shape==future.shape and torch.isfinite(loss)
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
torch.cuda.synchronize()
record={'seed':0,'shape':list(out.shape),'device':torch.cuda.get_device_name(),'python':sys.version.split()[0],'torch':torch.__version__,'loss':loss.item(),'finite_gradients':True,'configured_cuda_fraction':.65,'params':sum(p.numel() for p in model.parameters()),'purpose':'environment test only, synthetic inputs'}
print('TEXTOP_KERNEL_WITNESS',json.dumps(record),flush=True)
root=pathlib.Path(__file__).resolve().parents[1]
(root/'receipts/kernel_witness.json').write_text(json.dumps(record,indent=2))

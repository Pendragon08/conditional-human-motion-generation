"""Seeded real kernels in the two already-installed generation runtimes."""
import argparse,json,sys
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('--mode',choices=['generation','retarget'],required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists()
torch.manual_seed(20261007)
x=torch.randn(8,8,device='cuda',requires_grad=True)
loss=(x@x.T).square().mean();loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all() and x.grad.is_cuda
result={'mode':a.mode,'python':sys.version,'torch':torch.__version__,'CUDA':torch.version.cuda,
        'device':torch.cuda.get_device_name(),'torch_loss':float(loss),'finite_cuda_backward':True}
if a.mode=='retarget':
    import jax,jaxlib,jax.numpy as jnp,numpy as np
    y=jax.random.normal(jax.random.PRNGKey(20261007),(8,8))
    z=jax.jit(lambda v:(v@v.T).sum())(y);z.block_until_ready()
    assert np.isfinite(np.asarray(z)) and y.devices().pop().platform=='gpu'
    result.update(jax=jax.__version__,jaxlib=jaxlib.__version__,numpy=np.__version__,jax_value=float(z),jax_gpu_kernel=True)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print('WORLD_CONFIRMATION_RUNTIME_WITNESS',a.mode,result,flush=True)

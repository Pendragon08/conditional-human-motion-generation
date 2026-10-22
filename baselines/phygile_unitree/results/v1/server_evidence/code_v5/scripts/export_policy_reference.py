"""Export original-runtime CPU/GPU numeric references, without simulation/RL."""
from pathlib import Path
import argparse,hashlib,importlib.util,inspect,json,sys
import numpy as np
import torch
from rsl_rl.modules import EmpiricalNormalization

p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
c=json.loads((a.assets/'contract.json').read_text());cp=a.assets/'parent_checkpoint.pt'
payload=torch.load(cp,map_location='cpu',weights_only=False)
spec=importlib.util.spec_from_file_location('original_encoder',a.assets/'encoder_actor_critic.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
norm=EmpiricalNormalization(shape=[616]).eval();norm.load_state_dict(payload['obs_norm_state_dict'],strict=True)
g=torch.Generator().manual_seed(42)
with np.load(a.assets/'synthetic_policy_fixture.npz',allow_pickle=False) as d:raw=torch.tensor(d['observations'])
bounded=norm._mean+(norm._std+1e-2)*torch.randn(5,616,generator=g)
arrays={'raw_observations':raw.numpy(),'bounded_observations':bounded.numpy(),'mean':norm._mean.numpy(),'std':norm._std.numpy()}
cfg=dict(c['policy_config']);cfg.pop('class_name')
with torch.inference_mode():
 for device,label in [('cpu','cpu'),('cuda:1','gpu_tf32_off')]:
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  model=module.MoEActorCritic(num_actor_obs=616,num_critic_obs=748,num_actions=29,**cfg).to(device).eval()
  model.load_state_dict(payload['model_state_dict'],strict=True);model._max_active_experts=c['active_experts'];norm=norm.to(device)
  for key,x in [('raw',raw),('bounded',bounded)]:
   arrays[key+'_'+label]=model.act_inference(norm(x.to(device))).cpu().numpy()
   arrays[key+'_normalized_'+label]=norm(x.to(device)).cpu().numpy()
np.savez_compressed(a.output/'reference.npz',**arrays)
(a.output/'normalizer_source.txt').write_text(inspect.getsource(EmpiricalNormalization))
(a.output/'receipt.json').write_text(json.dumps({'status':'REFERENCE_EXPORTED','torch':torch.__version__,'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'normalizer_source_sha256':hashlib.sha256(inspect.getsource(EmpiricalNormalization).encode()).hexdigest(),'scope':'Synthetic tensor reference only; not tracking performance.','gpu_tf32':False},indent=2)+'\n')
print('REFERENCE_EXPORTED',flush=True)

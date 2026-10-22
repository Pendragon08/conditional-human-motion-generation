"""Report every original and portable parity comparison, retaining raw failures."""
from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np
import torch
from frozen_policy import FrozenPhyGile
p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4);c=json.loads((a.assets/'contract.json').read_text(encoding='utf-8'))
model=FrozenPhyGile(a.assets/'parent_checkpoint.pt',c,a.assets/'encoder_actor_critic.py')
rows={}
with np.load(a.assets/'numeric_reference_v1.npz',allow_pickle=False) as d:
 for kind in ['raw','bounded']:
  obs=torch.tensor(d[kind+'_observations']);actual=model(obs).numpy()
  normalized=((obs-model.norm['_mean'])/(model.norm['_std']+1e-2)).numpy()
  for device in ['cpu','gpu_tf32_off']:
   expected=d[kind+'_'+device]
   rows[kind+'_portable_vs_'+device]={'max_abs_action_error':float(np.abs(actual-expected).max()),'allclose':bool(np.allclose(actual,expected,rtol=1e-5,atol=1e-5)),
    'max_abs_normalized_obs_error':float(np.abs(normalized-d[kind+'_normalized_'+device]).max()),'max_abs_normalized_observation':float(np.abs(normalized).max())}
  rows[kind+'_original_cpu_vs_gpu']={'max_abs_action_error':float(np.abs(d[kind+'_cpu']-d[kind+'_gpu_tf32_off']).max()),'allclose':bool(np.allclose(d[kind+'_cpu'],d[kind+'_gpu_tf32_off'],rtol=1e-5,atol=1e-5))}
result={'status':'BOUNDED_NORMALIZED_INPUT_PARITY_PASSED_RAW_INPUT_WARNING' if all(rows['bounded_portable_vs_'+d]['allclose'] for d in ['cpu','gpu_tf32_off']) else 'PARITY_FAILED',
 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rtol':1e-5,'atol':1e-5,'comparisons':rows,'torch':torch.__version__,'checkpoint_sha256':hashlib.sha256((a.assets/'parent_checkpoint.pt').read_bytes()).hexdigest(),
 'scope':'Two five-row synthetic fixtures; bounded input uses original empirical mean/std plus seeded unit Gaussian. Raw fixture and failures retained. No performance result.'}
with a.output.open('x',encoding='utf-8') as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
if result['status']=='PARITY_FAILED':raise SystemExit(1)

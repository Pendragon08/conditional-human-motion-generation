"""Original-runtime FP32 parity and retained original TF32 discrepancy."""
from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np
import torch
from frozen_policy import FrozenPhyGile
p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4);c=json.loads((a.assets/'contract.json').read_text(encoding='utf-8'))
model=FrozenPhyGile(a.assets/'parent_checkpoint.pt',c,a.assets/'encoder_actor_critic.py');rows={}
with np.load(a.assets/'numeric_reference_v2.npz',allow_pickle=False) as d,np.load(a.assets/'synthetic_policy_fixture.npz',allow_pickle=False) as old:
 for kind in ['raw','bounded']:
  obs=torch.tensor(d[kind+'_observations']);actual=model(obs).numpy();normalized=((obs-model.norm['_mean'])/(model.norm['_std']+1e-2)).numpy()
  for device in ['cpu','gpu_tf32_off','gpu_tf32_on']:
   expected=d[kind+'_'+device]
   rows[kind+'_portable_vs_'+device]={'max_abs_action_error':float(np.abs(actual-expected).max()),'allclose':bool(np.allclose(actual,expected,rtol=1e-5,atol=1e-5)),
    'max_abs_normalized_obs_error':float(np.abs(normalized-d[kind+'_normalized_'+device]).max()),'max_abs_normalized_observation':float(np.abs(normalized).max())}
 rows['initial_gpu_fixture_vs_tf32_on']={'max_abs_action_error':float(np.abs(old['actions']-d['raw_gpu_tf32_on']).max()),'observations_exact':bool(np.array_equal(old['observations'],d['raw_observations']))}
fp32_pass=all(rows[k+'_portable_vs_'+d]['allclose'] for k in ['raw','bounded'] for d in ['cpu','gpu_tf32_off'])
result={'status':'FP32_PARITY_PASSED_TF32_DIFFERENCE_RETAINED' if fp32_pass else 'FP32_PARITY_FAILED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rtol':1e-5,'atol':1e-5,
 'comparisons':rows,'torch':torch.__version__,'checkpoint_sha256':hashlib.sha256((a.assets/'parent_checkpoint.pt').read_bytes()).hexdigest(),
 'scope':'Two five-row synthetic fixtures, same original model/state/normalizer; FP32 CPU/GPU no-TF32 comparisons. Initial TF32 fixture failure retained. Not tracking performance.'}
with a.output.open('x',encoding='utf-8') as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
if not fp32_pass:raise SystemExit(1)

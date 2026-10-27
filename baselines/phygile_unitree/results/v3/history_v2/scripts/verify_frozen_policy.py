"""Compare original GPU policy actions to the portable frozen inference path."""
from pathlib import Path
import argparse,hashlib,json,datetime,os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import numpy as np,torch
from frozen_policy import FrozenPhyGile
p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4)
c=json.loads((a.assets/'contract.json').read_text(encoding='utf-8'))
policy=FrozenPhyGile(a.assets/'parent_checkpoint.pt',c,a.assets/'encoder_actor_critic.py')
with np.load(a.assets/'synthetic_policy_fixture.npz',allow_pickle=False) as data:
 obs=data['observations'];expected=data['actions']
 actual=policy(torch.tensor(obs)).numpy()
error=float(np.max(np.abs(expected-actual)))
if not np.allclose(actual,expected,rtol=1e-5,atol=1e-5):raise RuntimeError('Portable policy differs from original-runtime fixture: '+str(error))
result={'status':'FROZEN_POLICY_NUMERIC_PARITY_PASSED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'max_abs_action_error':error,'rtol':1e-5,'atol':1e-5,'fixture_shape':list(obs.shape),'action_shape':list(actual.shape),'torch':torch.__version__,'checkpoint_sha256':hashlib.sha256((a.assets/'parent_checkpoint.pt').read_bytes()).hexdigest(),'scope':'Synthetic random observation tensor parity only; not a tracking benchmark.'}
with a.output.open('x',encoding='utf-8') as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

"""Fresh current exporter and actual production compact-loader witness."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--scripts',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);archive=a.output/'source_snapshot';archive.mkdir()
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
files=[a.scripts/n for n in ['export_active_policy.py','frozen_policy.py','active_policy_artifact.py']]+[Path(__file__),a.assets/'encoder_actor_critic.py',a.assets/'contract.json']
source={str(f):sha(f) for f in files}
for f in files:shutil.copyfile(f,archive/f.name)
command=[sys.executable,str(a.scripts/'export_active_policy.py'),'--checkpoint',str(a.assets/'parent_checkpoint.pt'),'--contract',str(a.assets/'contract.json'),'--encoder-source',str(a.assets/'encoder_actor_critic.py'),'--output',str(a.output/'export')]
with (a.output/'export_stdout.txt').open('x',encoding='utf-8') as so,(a.output/'export_stderr.txt').open('x',encoding='utf-8') as se:r=subprocess.run(command,stdout=so,stderr=se)
if r.returncode:raise RuntimeError('Current exporter did not complete')
sys.path.insert(0,str(a.scripts))
import torch,numpy as np
from frozen_policy import FrozenPhyGile
torch.set_num_threads(4);torch.manual_seed(9876)
c=json.loads((a.assets/'contract.json').read_text(encoding='utf-8'));full=FrozenPhyGile(a.assets/'parent_checkpoint.pt',c,a.assets/'encoder_actor_critic.py')
compact=FrozenPhyGile(a.output/'export/active_policy.pt',c,a.assets/'encoder_actor_critic.py')
fixture=np.load(a.output/'export/synthetic_inference_fixture.npz',allow_pickle=False)
obs=torch.from_numpy(fixture['observations']);expected=torch.from_numpy(fixture['actions']);checks=[]
for batch in [1,7,128,1024]:
 x=obs[:batch];actual=compact(x);original=full(x)
 if not torch.equal(actual,original) or not torch.equal(actual,expected[:batch]):raise AssertionError('Production loader changed same-input actions')
 checks.append({'batch':batch,'bitwise_equal':True,'max_abs_action_delta':float((actual-original).abs().max())})
for f,d in source.items():assert sha(f)==d,'Source changed during execution'
payload=torch.load(a.output/'export/active_policy.pt',map_location='cpu',weights_only=False)
assert len(compact.model.critic._forward_pre_hooks)>=1 and all(len(x._forward_pre_hooks)>=1 for x in compact.model.experts[1:])
proof={'status':'CURRENT_EXPORT_AND_PRODUCTION_LOADER_BITWISE_PARITY','export_exit_code':r.returncode,'command':command,
 'production_loader':'frozen_policy.FrozenPhyGile -> active_policy_artifact.load_active','batch_checks':checks,
 'artifact_sha256':sha(a.output/'export/active_policy.pt'),'source_checkpoint_sha256':sha(a.assets/'parent_checkpoint.pt'),
 'dropped_branches_guarded':True,'optimizer_or_physics_updates':0,'source_files_sha256':source,
 'executed_sources_unchanged':True,'source_snapshot_archived_before_execution':True,'evaluation_type':'synthetic_proxy',
 'scope':'Same-input FP32 inference parity only; not motion quality or training resume from compact artifact'}
(a.output/'production_proof.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');print(json.dumps(proof))

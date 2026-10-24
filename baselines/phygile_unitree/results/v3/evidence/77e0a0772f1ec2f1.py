"""Export and prove active-branch parity against the full checkpoint on CPU."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import torch
from frozen_policy import FrozenPhyGile
from active_policy_artifact import KIND,omitted_key,load_active
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--encoder-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(42)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=json.loads(a.contract.read_text(encoding='utf-8'));assert c['active_experts']==1
assert c['checkpoint_sha256']==sha(a.checkpoint),'Use the matching deterministic evaluation contract'
payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
assert 'optimizer_state_dict' in payload,'Source must be a full resume checkpoint'
full=FrozenPhyGile(a.checkpoint,c,a.encoder_source,'cpu')
active={'artifact_kind':KIND,'active_experts':1,'source_checkpoint_sha256':sha(a.checkpoint),
 'encoder_source_sha256':sha(a.encoder_source),'contract_sha256':sha(a.contract),'source_iteration':payload['iter'],
 'model_state_dict':{k:v for k,v in payload['model_state_dict'].items() if not omitted_key(k)},
 'obs_norm_state_dict':payload['obs_norm_state_dict'],'omitted_state_keys':sorted(k for k in payload['model_state_dict'] if omitted_key(k)),
 'training_resume_supported':False,'precision':'Original FP32 tensors; no quantization or reparameterization'}
dest=a.output/'active_policy.pt';torch.save(active,dest)
# Independently initialize discarded branches; hooks forbid using any of them.
torch.manual_seed(9876);compact=FrozenPhyGile(a.checkpoint,c,a.encoder_source,'cpu')
for name,value in compact.model.state_dict().items():
 if omitted_key(name) and value.is_floating_point():value.normal_()
load_active(compact.model,torch.load(dest,map_location='cpu',weights_only=False),c)
generator=torch.Generator().manual_seed(2026);observations=torch.randn((1024,616),generator=generator)
with torch.inference_mode():
 expected=full(observations);actual=compact(observations)
if not torch.equal(expected,actual):raise AssertionError('Active artifact is not bitwise equal on the same CPU input')
np.savez_compressed(a.output/'synthetic_inference_fixture.npz',observations=observations.numpy(),actions=expected.numpy())
proof={'status':'BITWISE_ACTIVE_INFERENCE_PARITY','artifact_kind':KIND,'source_checkpoint_sha256':sha(a.checkpoint),
 'artifact_sha256':sha(dest),'artifact_bytes':dest.stat().st_size,'source_checkpoint_bytes':a.checkpoint.stat().st_size,
 'active_experts':1,'omitted_state_keys':active['omitted_state_keys'],'synthetic_rows':1024,'seed':2026,
 'max_abs_action_delta':float((expected-actual).abs().max()),'discarded_branches_randomized_and_forbidden':True,
 'training_resume_supported':False,'full_checkpoint_preserved':True,'source_files_sha256':{str(x):sha(x) for x in [Path(__file__),Path(__file__).with_name('active_policy_artifact.py'),Path(__file__).with_name('frozen_policy.py'),a.encoder_source]}}
(a.output/'artifact_proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))

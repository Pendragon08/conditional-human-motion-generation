"""Read-only identity/schema checks against a preserved full resume checkpoint."""
from pathlib import Path
import argparse,hashlib,json
import torch
import active_policy_artifact
from active_policy_artifact import KIND,omitted_key
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--encoder-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists();torch.set_num_threads(4)
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
before={str(f):sha(f) for f in [a.checkpoint,a.artifact,a.contract,a.encoder_source,Path(__file__),Path(active_policy_artifact.__file__)]}
full=torch.load(a.checkpoint,map_location='cpu',weights_only=False);artifact=torch.load(a.artifact,map_location='cpu',weights_only=False);c=json.loads(a.contract.read_text(encoding='utf-8'))
assert artifact['artifact_kind']==KIND and artifact['active_experts']==c['active_experts']==1
assert artifact['source_checkpoint_sha256']==sha(a.checkpoint)==c['checkpoint_sha256']
assert artifact['encoder_source_sha256']==sha(a.encoder_source)
assert artifact['contract_sha256']==sha(a.contract)
expected_omitted={k for k in full['model_state_dict'] if omitted_key(k)}
assert set(artifact['omitted_state_keys'])==expected_omitted
assert set(artifact['model_state_dict'])==set(full['model_state_dict'])-expected_omitted
assert 'optimizer_state_dict' not in artifact and artifact['training_resume_supported'] is False
for name,value in artifact['model_state_dict'].items():
 original=full['model_state_dict'][name];assert value.dtype==original.dtype and value.shape==original.shape and torch.equal(value,original)
 assert bool(torch.isfinite(value).all())
norm=artifact['obs_norm_state_dict'];original=full['obs_norm_state_dict']
assert set(norm)==set(original)=={'_mean','_var','_std','count'}
rows={}
for name,value in norm.items():
 reference=original[name]
 assert value.shape==reference.shape and value.dtype==reference.dtype and torch.equal(value,reference)
 assert bool(torch.isfinite(value).all())
 if name=='count':assert value.shape==torch.Size([]) and value.dtype==torch.int64 and int(value)>0
 else:assert tuple(value.shape)==(1,616) and value.dtype==torch.float32
 rows[name]={'shape':list(value.shape),'dtype':str(value.dtype),'value_sha256':hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()}
assert bool((norm['_std']>=0).all()) and bool((norm['_var']>=0).all())
assert all(bool(torch.isfinite(v).all()) for v in full['model_state_dict'].values())
assert all(bool(torch.isfinite(v).all()) for state in full['optimizer_state_dict']['state'].values() for v in state.values() if isinstance(v,torch.Tensor))
for f,d in before.items():assert sha(f)==d
result={'status':'FULL_CHECKPOINT_ARTIFACT_CONTRACT_ENCODER_NORMALIZER_BOUND','source_checkpoint_sha256':sha(a.checkpoint),
 'artifact_sha256':sha(a.artifact),'active_experts':1,'retained_model_tensors_exact':len(artifact['model_state_dict']),
 'omitted_model_tensors':len(expected_omitted),'normalizer_schema':rows,'all_full_model_and_optimizer_tensors_finite':True,
 'normalizer_and_retained_state_bitwise_equal':True,'training_resume_supported':False,'inputs_unchanged':True,
 'source_and_input_hashes':before,'scope':'Read-only artifact identity and numerical state binding; complements separately witnessed production inference. No training or motion-quality result.'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

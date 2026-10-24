from pathlib import Path
import base64,datetime,hashlib,json
import torch
torch.set_num_threads(4)
root=Path('/data/code/ouyanghao/oyh/phygile-unitree-20261022')
code=root/'code_v9';run=root/'runs/headless_smoke_v3'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
receipt=json.loads((run/'receipt.json').read_text())
manifest=json.loads((run/'executed_sources.json').read_text())
changed=[];missing=[]
for name,digest in manifest['files'].items():
 p=Path(name)
 if not p.is_file():missing.append(name)
 elif sha(p)!=digest:changed.append(name)
baseline_changed=[];baseline_missing=[]
for name,data in EXPECTED['source_manifest'].items():
 p=code/name
 if not p.is_file():baseline_missing.append(name)
 elif sha(p)!=data['sha256']:baseline_changed.append(name)
parent=Path(receipt['parent_checkpoint']);final=Path(receipt['final_checkpoint'])
parent_ckpt=torch.load(parent,map_location='cpu',weights_only=False)
final_ckpt=torch.load(final,map_location='cpu',weights_only=False)
def tensors(value,path=''):
 if isinstance(value,torch.Tensor):yield path,value
 elif isinstance(value,dict):
  for k,v in value.items():yield from tensors(v,path+'/'+str(k))
 elif isinstance(value,(tuple,list)):
  for k,v in enumerate(value):yield from tensors(v,path+'/'+str(k))
changed_model=[k for k,v in parent_ckpt['model_state_dict'].items() if not torch.equal(v,final_ckpt['model_state_dict'][k])]
nonfinite_model=[k for k,v in tensors(final_ckpt['model_state_dict']) if not bool(torch.isfinite(v).all())]
nonfinite_optimizer=[k for k,v in tensors(final_ckpt['optimizer_state_dict']) if not bool(torch.isfinite(v).all())]
blobs={}
for name in ['scripts/lab_run.py','scripts/source_provenance.py','scripts/run_isaac.sh']:
 p=code/name;data=p.read_bytes()
 blobs[name]={'sha256':hashlib.sha256(data).hexdigest(),'size':len(data),'base64':base64.b64encode(data).decode()}
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only_audit':True,
'receipt_status':receipt['status'],'manifest_sha256':sha(run/'executed_sources.json'),
'manifest_sha_matches_receipt':sha(run/'executed_sources.json')==receipt['source_provenance']['sha256'],
'manifest_file_count':len(manifest['files']),'manifest_post_exit_changed':changed,'manifest_post_exit_missing':missing,
'preexisting_code_files_checked':len(EXPECTED['source_manifest']),'preexisting_code_changed':baseline_changed,'preexisting_code_missing':baseline_missing,
'parent_sha256_before':EXPECTED['parent']['sha256'],'parent_sha256_after':sha(parent),'parent_unchanged_independent':sha(parent)==EXPECTED['parent']['sha256']==receipt['parent_sha256'],
'final_sha256':sha(final),'final_sha_matches_receipt':sha(final)==receipt['final_sha256'],
'parent_checkpoint_iter':parent_ckpt.get('iter'),'final_checkpoint_iter':final_ckpt.get('iter'),
'model_keys_equal':parent_ckpt['model_state_dict'].keys()==final_ckpt['model_state_dict'].keys(),
'changed_model_state_tensors':changed_model,'changed_model_state_tensor_count':len(changed_model),
'final_model_tensor_count':sum(1 for _ in tensors(final_ckpt['model_state_dict'])),'nonfinite_model_tensors':nonfinite_model,
'final_optimizer_tensor_count':sum(1 for _ in tensors(final_ckpt['optimizer_state_dict'])),'nonfinite_optimizer_tensors':nonfinite_optimizer,
'parent_adam_steps':sorted({float(s['step']) for s in parent_ckpt['optimizer_state_dict']['state'].values() if 'step' in s}),
'final_adam_steps':sorted({float(s['step']) for s in final_ckpt['optimizer_state_dict']['state'].values() if 'step' in s}),
'source_blobs':blobs}
print(json.dumps(result,indent=2))


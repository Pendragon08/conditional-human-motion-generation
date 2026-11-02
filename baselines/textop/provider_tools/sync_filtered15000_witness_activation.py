"""Adopt only actually closed CPU acceptance into local VAE witness bindings."""
import pathlib,json,hashlib,base64,shutil
B=pathlib.Path(__file__).resolve().parents[1];D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
O=B/'results/filtered15000_handoff_20261027';p=O/'filtered15000_acceptance_v1/progress.json';j=json.loads(p.read_text(encoding='utf-8'))
assert j['kind']=='accept_data' and j['closed_execution_verified'] and j['common_lock_free'] and not j['workers']
r=j['files']['receipt'];assert r['status']=='ACTUAL_FILTERED_CORPUS_VAE_WITNESSES_ENABLED_NOT_DISPATCHED'
assert j['files']['outer']['actual_returncode']==0 and r['output_writers_closed']
raw=base64.b64decode(r['new_vae_spec_base64']);assert hashlib.sha256(raw).hexdigest()==r['new_vae_spec_sha256'];s=json.loads(raw)
assert s['schema']=='textop.filtered15000_vae.v1' and s['witness_authorized'] and not s['train_authorized'] and s['witnesses']==[]
assert s['data_acceptance_sha256']==r['data_acceptance_sha256']
accepted_bytes=base64.b64decode(r['data_acceptance_base64'])
assert hashlib.sha256(accepted_bytes).hexdigest()==r['data_acceptance_sha256'] and json.loads(accepted_bytes)==r['data_acceptance']
assert s['checkpoint_allowed_root']=='/home/ouyanghao/textop-filtered15000-checkpoints-20261027'
for name,h in s['bound_sources'].items():assert hashlib.sha256((B/name).read_bytes()).hexdigest()==h,name
assert hashlib.sha256((B/'scripts/run_filtered15000_vae.py').read_bytes()).hexdigest()==s['entrypoint_sha256']
sp=B/'configs/filtered15000-vae-spec.json';old=json.loads(sp.read_text(encoding='utf-8'));assert not old['witness_authorized'] and old['data_acceptance_sha256'] is None
ap=O/'data_acceptance.json';assert not ap.exists()
sp.write_bytes(raw);ap.write_bytes(accepted_bytes)
for src in [sp,ap]:
 dst=D/src.relative_to(B);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert src.read_bytes()==dst.read_bytes()
print(json.dumps(dict(status='ACTUAL_ACCEPTANCE_LOCAL_WITNESS_BINDINGS_ADOPTED',vae_witness_authorized=True,formal_train_authorized=False,new_training_runs=0)))

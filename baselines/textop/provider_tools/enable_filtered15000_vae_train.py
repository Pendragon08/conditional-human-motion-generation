"""Enable new VAE200k after two actual bound gradient/restore witnesses."""
import argparse,pathlib,json,hashlib,base64,ast
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--document-summary',required=True);a=p.parse_args()
dp=pathlib.Path(a.document_summary).resolve();doc=json.loads(dp.read_text(encoding='utf-8'))
assert doc['status']=='FILTERED15000_VAE_FRESH_DOCUMENT_VERIFIED' and doc['run']=='filtered15000_vae_document_v1' and doc['actual_exit_code']==0 and doc['findings']==[]
sp=B/'configs/filtered15000-vae-spec.json';old=sp.read_bytes();s=json.loads(old)
assert s['witness_authorized'] and not s['train_authorized'] and s['witnesses']==[]
keys=['vae','seed','corpus_receipt_sha256','data_acceptance_sha256','strict_audit_receipt_sha256','bound_sources','entrypoint_sha256','checkpoint_dir','checkpoint_allowed_root','checkpoint_policy']
signature=hashlib.sha256(json.dumps({k:s[k] for k in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert doc['runtime_binding_sha256']==signature
witnesses=[]
for run in ['filtered15000_vae_author_v1','filtered15000_vae_document_v1']:
 j=json.loads((B/'results/filtered15000_handoff_20261027'/run/'progress.json').read_text(encoding='utf-8'))
 assert j['kind']=='vae' and j['closed_execution_verified'] and not j['workers']
 r=j['files']['receipt'];assert r['status']=='FILTERED15000_VAE_WITNESS_PASSED' and r['mode']=='witness'
 assert r['runtime_binding_sha256']==signature and r['source_sha256']==s['entrypoint_sha256']
 assert r['completed_primitive_attempts']==r['optimizer_updates']==4 and r['skipped_updates']==0
 m=r['memory_witness'];assert m['all_four_training_gradients_finite'] and m['model_and_adam_roundtrip_exact'] and m['next_batch_and_rng_restored'] and m['all_rng_recursive_equal']
 assert m['validation_optimizer_updates']==0 and m['pending_evaluation_completed_without_updates'] and not m['weights_reused_for_training']
 witnesses.append(dict(run=run,receipt_sha256=j['sha256']['receipt'],exit_sha256=j['sha256']['exit']))
assert j['common_lock_free'];s.update(train_authorized=True,witnesses=witnesses,independent_document_summary_sha256=hashlib.sha256(dp.read_bytes()).hexdigest())
raw=(json.dumps(s,indent=2)+'\n').encode();expected=hashlib.sha256(old).hexdigest()
code=r'''
import pathlib,json,hashlib,base64,os,sys,fcntl,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
p=r/'configs/filtered15000-vae-spec.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==__OLD_SPEC__
data=base64.b64decode(__NEW_BYTES__);s=json.loads(data);sys.path.insert(0,str(r/'scripts'))
from filtered15000_corpus_acceptance import read,sha
from filtered15000_inputs import load_selection
from checkpoint_storage_guard import check
load_selection(read(r/'configs/filtered15000-corpus-spec.json'))
ap=r/'receipts/filtered15000_data_acceptance_v1.json';assert sha(ap)==s['data_acceptance_sha256'];acc=read(ap)
assert acc['actual_strict_filtered_training_data_accepted'] and acc['private_server_D_backup_receipt_sha256'] and not acc['paper_two_stage_route_reproduced']
keys=['vae','seed','corpus_receipt_sha256','data_acceptance_sha256','strict_audit_receipt_sha256','bound_sources','entrypoint_sha256','checkpoint_dir','checkpoint_allowed_root','checkpoint_policy']
signature=hashlib.sha256(json.dumps({k:s[k] for k in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert signature==__NUMERIC_SIGNATURE__ and len({w['run'] for w in s['witnesses']})==2
for name,h in s['bound_sources'].items():assert sha(r/name)==h,name
assert sha(r/'scripts/run_filtered15000_vae.py')==s['entrypoint_sha256'] and sha(r/'scripts/supervise_filtered15000_vae.py')==s['supervisor_sha256']
for w in s['witnesses']:
 rp=r/'filtered15000_20261027/vae_runs'/w['run']/'receipt.json';prefix='filtered15000_vae_'+w['run'];ep=r/'receipts'/(prefix+'_exit.json')
 assert sha(rp)==w['receipt_sha256'] and sha(ep)==w['exit_sha256'];rec,e=read(rp),read(ep)
 assert e['evaluation_receipt_sha256']==sha(rp) and e['supervision_passed'] and e['child_reaped'] and e['child_exit_code']==0 and not e['timed_out']
 assert read(r/'receipts'/(prefix+'_outer_exit.json'))['actual_returncode']==0
 assert rec['status']=='FILTERED15000_VAE_WITNESS_PASSED' and rec['output_writers_closed'] and rec['runtime_binding_sha256']==signature
 assert rec['corpus_receipt_sha256']==s['corpus_receipt_sha256'] and rec['data_acceptance_sha256']==s['data_acceptance_sha256'] and rec['source_sha256']==s['entrypoint_sha256']
 m=rec['memory_witness'];assert m['all_four_training_gradients_finite'] and m['model_and_adam_roundtrip_exact'] and m['next_batch_and_rng_restored'] and m['all_rng_recursive_equal'] and m['validation_optimizer_updates']==0
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
dest=pathlib.Path(s['checkpoint_dir']);assert not list(dest.glob('*.pth')) and not list(dest.glob('*.partial'))
capacity=check(dest,pathlib.Path(s['checkpoint_allowed_root']),s['checkpoint_policy'],remaining_saves=5)
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()};assert free[1]>20480
readiness=dict(status='ACTUAL_FILTERED_CORPUS_AND_TWO_GRADIENT_RESTORE_WITNESSES_ACCEPTED',filtered_corpus_accepted=True,
 formal_training_ready=True,generator_started=False,runtime_binding_sha256=signature,checkpoint_capacity=capacity,
 data_acceptance_sha256=s['data_acceptance_sha256'],corpus_receipt_sha256=s['corpus_receipt_sha256'],
 witnesses=s['witnesses'],fresh_document_summary_sha256=s['independent_document_summary_sha256'],
 diagnostic_weights_reused=False,new_training_steps=0,paper_two_stage_route_reproduced=False)
with (r/'receipts/filtered15000_training_readiness_v1.json').open('x') as f:json.dump(readiness,f,indent=2)
with (r/'receipts/filtered15000_vae_spec_before_train.json').open('xb') as f:f.write(p.read_bytes())
temp=p.with_name(p.name+'.train_enable.partial')
with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
temp.replace(p);lock.close()
print(json.dumps(dict(status='NEW_FILTERED_VAE200K_ENABLED_NOT_DISPATCHED',new_spec_sha256=sha(p),readiness=readiness,new_gpu_contexts=0,new_model_loads=0)))
'''.replace('__OLD_SPEC__',repr(expected)).replace('__NEW_BYTES__',repr(base64.b64encode(raw).decode())).replace('__NUMERIC_SIGNATURE__',repr(signature))
ast.parse(code)
folder=B/'results/filtered15000_handoff_tools_20261027'
result=json.loads(recorded_remote(code,folder/'enable_vae_train_v1_client.json',timeout=90));sp.write_bytes(raw)
(folder/'vae_train_activation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

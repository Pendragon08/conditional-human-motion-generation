"""Enable new LDM300k after two actual bound gradient/restore witnesses."""
import argparse,pathlib,json,hashlib,base64,ast
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--document-summary',required=True);a=p.parse_args()
dp=pathlib.Path(a.document_summary).resolve();doc=json.loads(dp.read_text(encoding='utf-8'))
assert doc['status']=='USER_VAE100K_LDM_FRESH_DOCUMENT_VERIFIED' and doc['run']=='user_vae100k_ldm_document_v1' and doc['actual_exit_code']==0 and doc['findings']==[]
sp=B/'configs/user-vae100k-ldm-spec.json';old=sp.read_bytes();s=json.loads(old)
assert s['witness_authorized'] and not s['train_authorized'] and s['witnesses']==[]
keys=['ldm','seed','corpus_receipt_sha256','data_acceptance_sha256','strict_audit_receipt_sha256','vae_training','vae_export','vae_inference','bound_sources','entrypoint_sha256','checkpoint_dir','checkpoint_allowed_root','checkpoint_policy','checkpoint_layout']
signature=hashlib.sha256(json.dumps({k:s[k] for k in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert doc['runtime_binding_sha256']==signature
witnesses=[]
for run in ['user_vae100k_ldm_author_v1','user_vae100k_ldm_document_v1']:
 j=json.loads((B/'results/filtered15000_handoff_20261027'/run/'progress.json').read_text(encoding='utf-8'))
 assert j['kind']=='ldm' and j['closed_execution_verified'] and not j['workers']
 r=j['files']['receipt'];assert r['status']=='FILTERED15000_LDM_WITNESS_PASSED' and r['mode']=='witness'
 assert r['runtime_binding_sha256']==signature and r['source_sha256']==s['entrypoint_sha256']
 assert r['completed_primitive_attempts']==r['optimizer_updates']==4 and r['skipped_updates']==0
 m=r['memory_witness'];assert m['all_four_training_gradients_finite'] and m['model_and_adam_roundtrip_exact'] and m['next_batch_replayed_exactly'] and m['all_rngs_restored_exactly']
 assert m['validation_optimizer_updates']==0 and m['pending_evaluation_completed_without_updates'] and m['vae_frozen_and_unchanged'] and not m['weights_reused_for_training']
 h=r['full_history_helper_witness'];assert h['actual_denoiser_forward_calls']==5 and h['rng_restored'] and h['model_and_frozen_vae_unchanged'] and not h['formal_later_stage_training_reached']
 witnesses.append(dict(run=run,receipt_sha256=j['sha256']['receipt'],exit_sha256=j['sha256']['exit']))
assert j['common_lock_free'];s.update(train_authorized=True,witnesses=witnesses,independent_document_summary_sha256=hashlib.sha256(dp.read_bytes()).hexdigest())
raw=(json.dumps(s,indent=2)+'\n').encode();expected=hashlib.sha256(old).hexdigest()
code=r'''
import pathlib,json,hashlib,base64,os,sys,fcntl,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
p=r/'configs/user-vae100k-ldm-spec.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==__OLD_SPEC__
data=base64.b64decode(__NEW_BYTES__);s=json.loads(data);sys.path.insert(0,str(r/'scripts'))
from filtered15000_corpus_acceptance import read,sha
from filtered15000_inputs import load_selection
from user_vae100k_ldm_storage import check,contains_state_files
from user_vae100k_completion import accepted_vae100k as completed_vae,closed_job
load_selection(read(r/'configs/filtered15000-corpus-spec.json'));completed_vae(s['vae_training'])
ap=r/'receipts/filtered15000_data_acceptance_v1.json';assert sha(ap)==s['data_acceptance_sha256'];acc=read(ap)
assert acc['actual_strict_filtered_training_data_accepted'] and acc['private_server_D_backup_receipt_sha256'] and not acc['paper_two_stage_route_reproduced']
keys=['ldm','seed','corpus_receipt_sha256','data_acceptance_sha256','strict_audit_receipt_sha256','vae_training','vae_export','vae_inference','bound_sources','entrypoint_sha256','checkpoint_dir','checkpoint_allowed_root','checkpoint_policy','checkpoint_layout']
signature=hashlib.sha256(json.dumps({k:s[k] for k in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert signature==__NUMERIC_SIGNATURE__ and len({w['run'] for w in s['witnesses']})==2
for name,h in s['bound_sources'].items():assert sha(r/name)==h,name
assert sha(r/'scripts/run_user_vae100k_ldm.py')==s['entrypoint_sha256'] and sha(r/'scripts/supervise_user_vae100k_ldm.py')==s['supervisor_sha256']
for w in s['witnesses']:
 rp=r/'filtered15000_20261027/ldm_runs'/w['run']/'receipt.json';proof=closed_job('filtered15000_ldm_'+w['run'],rp,w['receipt_sha256'],s['supervisor_sha256']);assert proof['exit_sha256']==w['exit_sha256'];prefix='filtered15000_ldm_'+w['run'];ep=r/'receipts'/(prefix+'_exit.json')
 assert sha(rp)==w['receipt_sha256'] and sha(ep)==w['exit_sha256'];rec,e=read(rp),read(ep)
 assert e['evaluation_receipt_sha256']==sha(rp) and e['supervision_passed'] and e['child_reaped'] and e['child_exit_code']==0 and not e['timed_out']
 assert read(r/'receipts'/(prefix+'_outer_exit.json'))['actual_returncode']==0
 assert rec['status']=='FILTERED15000_LDM_WITNESS_PASSED' and rec['output_writers_closed'] and rec['runtime_binding_sha256']==signature
 assert rec['corpus_receipt_sha256']==s['corpus_receipt_sha256'] and rec['data_acceptance_sha256']==s['data_acceptance_sha256'] and rec['source_sha256']==s['entrypoint_sha256']
 m=rec['memory_witness'];assert m['all_four_training_gradients_finite'] and m['model_and_adam_roundtrip_exact'] and m['next_batch_replayed_exactly'] and m['all_rngs_restored_exactly'] and m['vae_frozen_and_unchanged'] and m['validation_optimizer_updates']==0
 h=rec['full_history_helper_witness'];assert h['actual_denoiser_forward_calls']==5 and h['rng_restored'] and h['model_and_frozen_vae_unchanged'] and not h['formal_later_stage_training_reached']
 assert rec['vae_inference_sha256']==s['vae_inference']['sha256']
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
dest=pathlib.Path(s['checkpoint_dir']);assert not contains_state_files(s);assert sha(dest/'vae.pth')==s['vae_inference']['sha256']
capacity=check(s)
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()};assert free[1]>20480
readiness=dict(status='ACTUAL_FILTERED_CORPUS_AND_TWO_GRADIENT_RESTORE_WITNESSES_ACCEPTED',filtered_corpus_accepted=True,
 formal_training_ready=True,generator_started=False,runtime_binding_sha256=signature,checkpoint_capacity=capacity,
 data_acceptance_sha256=s['data_acceptance_sha256'],corpus_receipt_sha256=s['corpus_receipt_sha256'],
 witnesses=s['witnesses'],fresh_document_summary_sha256=s['independent_document_summary_sha256'],
 vae_inference_sha256=s['vae_inference']['sha256'],real_full_history_helper_forward_calls_per_witness=5,formal_later_stage_training_reached=False,diagnostic_weights_reused=False,new_training_steps=0,paper_two_stage_route_reproduced=False)
with (r/'receipts/user_vae100k_ldm_training_readiness_v1.json').open('x') as f:json.dump(readiness,f,indent=2)
with (r/'receipts/user_vae100k_ldm_spec_before_train.json').open('xb') as f:f.write(p.read_bytes())
temp=p.with_name(p.name+'.train_enable.partial')
with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
temp.replace(p);lock.close()
print(json.dumps(dict(status='NEW_FILTERED_LDM300K_ENABLED_NOT_DISPATCHED',new_spec_sha256=sha(p),readiness=readiness,new_gpu_contexts=0,new_model_loads=0)))
'''.replace('__OLD_SPEC__',repr(expected)).replace('__NEW_BYTES__',repr(base64.b64encode(raw).decode())).replace('__NUMERIC_SIGNATURE__',repr(signature))
ast.parse(code)
folder=B/'results/filtered15000_handoff_tools_20261027'
result=json.loads(recorded_remote(code,folder/'enable_user_vae100k_ldm_train_v1_client.json',timeout=90));sp.write_bytes(raw)
(folder/'user_vae100k_ldm_train_activation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

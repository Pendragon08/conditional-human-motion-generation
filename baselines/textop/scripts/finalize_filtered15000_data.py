"""Bind positive private recovery+capacity to VAE witnesses; never train a model."""
import argparse,base64,json,os,pathlib,fcntl,shutil,subprocess
from filtered15000_corpus_acceptance import read,sha,owned,validate_corpus
from checkpoint_storage_guard import check
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--expected-vae-spec-sha',required=True);a=p.parse_args()
assert os.getuid()==1013 and a.run.isascii() and a.run.isidentifier()
ds=read(ROOT/'configs/filtered15000-recovery-spec.json');assert ds['prepare_authorized'] and ds['acceptance_authorized'] and sha(__file__)==ds['finalizer_sha256']
for n,h in ds['bound_sources'].items():assert sha(ROOT/n)==h,n
lock=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
out=ROOT/'filtered15000_20261027/data_runs'/a.run;rp=owned(out/'receipt.json',ROOT);r=read(rp)
prepared_spec_path=owned(ROOT/'receipts/filtered15000_recovery_spec_before_acceptance.json',ROOT)
prepared_spec=read(prepared_spec_path);expected_prepared=dict(ds);expected_prepared['acceptance_authorized']=False
assert prepared_spec==expected_prepared and prepared_spec['prepare_authorized'] and not prepared_spec['acceptance_authorized']
assert r['status']=='FILTERED15000_CORPUS_RECOVERY_PREPARED' and r['output_writers_closed'] and r['spec_sha256']==sha(prepared_spec_path)
assert r['source_sha256']==ds['executor_sha256']
prefix='filtered15000_recovery_'+a.run;ep=owned(ROOT/'receipts'/(prefix+'_exit.json'),ROOT);e=read(ep)
pp=owned(ROOT/'receipts'/(prefix+'_process_exit.json'),ROOT);pr=read(pp)
assert read(ROOT/'receipts'/(prefix+'_outer_exit.json'))['actual_returncode']==0
for item in [e,pr]:assert item['supervision_passed'] and item['child_reaped'] and item['child_exit_code']==0 and not item['timed_out'] and item['supervisor_error'] is None and item['sample_errors']==[] and not item.get('cleanup_error')
assert e['receipt_sha256']==sha(rp) and e['process_exit_sha256']==sha(pp)
assert e['spec_sha256']==sha(prepared_spec_path) and e['source_sha256']==ds['supervisor_sha256']
resources=owned(ROOT/'logs'/(prefix+'_resources.jsonl'),ROOT);assert sha(resources)==e['resource_sha256']
samples=[json.loads(x) for x in resources.read_text().splitlines()]
assert len(samples)==e['samples_written']==pr['samples_written'] and samples[-1]['phase']=='after_exit'
backup_path=owned(ROOT/'receipts/filtered15000_corpus_recovery_private_backup.json',ROOT);backup=read(backup_path)
arc=r['private_archive'];archive=owned(pathlib.Path(arc['path']),ROOT)
assert backup['status']=='CORPUS_RECOVERY_SERVER_D_SHA_VERIFIED' and backup['recovery_receipt_sha256']==sha(rp)
assert backup['archive_sha256']==arc['sha256']==sha(archive) and backup['archive_bytes']==arc['bytes']==archive.stat().st_size
assert backup['server_archive']==str(archive) and backup['actual_scp_exit_code']==0 and backup['D_owner_path_verified'] and backup['D_sha256']==arc['sha256']
assert backup['D_reserve_bytes_after']>=2*1024**3
checked=validate_corpus(ds,verify_artifacts=False);meta=checked['meta'];cs=checked['corpus_spec']
assert checked['view_receipt_sha256']==r['corpus_receipt_sha256'] and cs['strict_audit_receipt_sha256']==r['strict_audit_receipt_sha256']
sp=ROOT/'configs/filtered15000-vae-spec.json';assert sha(sp)==a.expected_vae_spec_sha;s=read(sp)
assert not s['witness_authorized'] and not s['train_authorized'] and s['witnesses']==[] and s['data_acceptance_sha256'] is None
assert sha(ROOT/'scripts/run_filtered15000_vae.py')==s['entrypoint_sha256'] and sha(ROOT/'scripts/supervise_filtered15000_vae.py')==s['supervisor_sha256']
for name,h in s['bound_sources'].items():assert sha(ROOT/name)==h,name
home=pathlib.Path('/home/ouyanghao');owned(home,home)
allowed=home/'textop-filtered15000-checkpoints-20261027';assert not allowed.exists() and not allowed.is_symlink()
required=s['checkpoint_policy']['reserve_bytes']+s['checkpoint_policy']['log_allowance_bytes']+5*s['checkpoint_policy']['checkpoint_upper_bound_bytes']
assert shutil.disk_usage(home).free>=required,'Preserve input and checkpoints; insufficient real HOME capacity'
assert shutil.disk_usage(ROOT).free>=2*1024**3
mem=[int(x.split()[1])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0];assert mem>=16*1024**3
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()};assert free[1]>20480
allowed.mkdir(mode=0o700);(allowed/'vae').mkdir(mode=0o700);dest=allowed/'vae/checkpoints';dest.mkdir(mode=0o700)
capacity=check(dest,allowed,s['checkpoint_policy'],remaining_saves=5)
acceptance=dict(schema='textop.filtered15000_loader_recovery_acceptance.v1',status='REAL_STRICT_CORPUS_LOADER_RECOVERY_AND_CAPACITY_ACCEPTED',
 actual_strict_filtered_training_data_accepted=True,paper_two_stage_route_reproduced=False,
 filtered_corpus_accepted=False,formal_gradient_witnesses_accepted=False,formal_training_ready=False,generator_started=False,
 strict_audit_receipt_sha256=cs['strict_audit_receipt_sha256'],corpus_spec_sha256=ds['corpus_spec_sha256'],
 corpus_receipt_sha256=checked['view_receipt_sha256'],source_sequences=meta['source_sequences'],source_sequences_in_view=meta['source_sequences_in_view'],
 motion_variants=meta['motion_variants'],input_library_sources=8278,actual_loader_witnesses=checked['witnesses'],
 fresh_document_summary_sha256=ds['fresh_document_summary_sha256'],private_recovery_receipt_sha256=sha(rp),
 private_recovery_exit_sha256=sha(ep),private_server_D_backup_receipt_sha256=sha(backup_path),
 original_array_assets_preserved=True,derived_RAM_cache_rebuild_recorded=True,train_only_69d_statistics=True,
 exact_cross_split_pose_overlap=0,complete_real_clip_embeddings=True,checkpoint_capacity=capacity,
 new_gpu_contexts=0,new_model_loads=0,new_sdk_runs=0,optimizer_updates=0,source_sha256=sha(__file__))
ap=ROOT/'receipts/filtered15000_data_acceptance_v1.json';assert not ap.exists()
archive_spec=ROOT/'receipts/filtered15000_vae_spec_before_witnesses.json'
with archive_spec.open('xb') as f:f.write(sp.read_bytes())
with ap.open('x') as f:json.dump(acceptance,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
s.update(data_acceptance_sha256=sha(ap),corpus_receipt_sha256=checked['view_receipt_sha256'],strict_audit_receipt_sha256=cs['strict_audit_receipt_sha256'],
 checkpoint_allowed_root=str(allowed),checkpoint_dir=str(dest),witness_authorized=True,train_authorized=False,witnesses=[])
data=(json.dumps(s,indent=2)+'\n').encode();temp=sp.with_name(sp.name+'.witness_enable.partial')
with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
temp.replace(sp)
result=dict(status='ACTUAL_FILTERED_CORPUS_VAE_WITNESSES_ENABLED_NOT_DISPATCHED',data_acceptance=acceptance,
 data_acceptance_sha256=sha(ap),data_acceptance_base64=base64.b64encode(ap.read_bytes()).decode(),
 new_vae_spec_sha256=sha(sp),new_vae_spec_base64=base64.b64encode(data).decode(),
 actual_new_training_started=False,new_model_loads=0,new_gpu_contexts=0,output_writers_closed=True)
with (ROOT/'receipts/filtered15000_vae_witness_activation.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)

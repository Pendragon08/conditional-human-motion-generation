"""Enable one explicit user100k successor stage, from actual closed upstream proof."""
import argparse,ast,base64,hashlib,json,pathlib,shutil
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1];O=B/'results/vae100k_user_handoff_20261028'
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['quality','export','witnesses'],required=True);a=p.parse_args()
names={'quality':'user-vae100k-quality-spec.json','export':'user-vae100k-export-spec.json','witnesses':'user-vae100k-ldm-spec.json'}
name=names[a.stage];sp=B/'configs'/name;old=sp.read_bytes();s=json.loads(old)
assert not any(s.get(k,False) for k in ['quality_authorized','export_authorized','witness_authorized','train_authorized'])
control=json.loads((O/'user_vae100k_control_v1/progress.json').read_bytes())
assert control['closed_execution_verified'] and control['files']['receipt']['selected_checkpoint_attempts']==100000
binding=dict(kind='USER_REQUESTED_VAE100000_PREFIX',selected_step=100000,control_run='user_vae100k_control_v1',control_receipt_sha256=control['sha256']['receipt'],control_exit_sha256=control['sha256']['exit'])
prior=control
if a.stage=='quality':
 backup=json.loads((O/'checkpoint100000_backup/backup.json').read_bytes())
 assert backup['status']=='USER_VAE100000_PRIVATE_BYTE_BACKUP_VERIFIED' and backup['control_binding']==binding
 assert backup['checkpoint']==control['files']['receipt']['selected_checkpoint'] and backup['D_sha256']==backup['checkpoint']['sha256']
else:
 quality=json.loads((O/'user_vae100k_quality_v1/progress.json').read_bytes());prior=quality
 assert quality['closed_execution_verified'] and quality['files']['receipt']['status']=='USER_VAE100K_QUALITY_VALIDATION_PASSED'
 binding.update(quality_run='user_vae100k_quality_v1',quality_receipt_sha256=quality['sha256']['receipt'],quality_exit_sha256=quality['sha256']['exit'])
s['vae_training']=binding
if a.stage=='witnesses':
 ex=json.loads((B/'results/filtered15000_handoff_20261027/user_vae100k_export_v1/progress.json').read_bytes());prior=ex
 assert ex['closed_execution_verified'] and ex['files']['receipt']['status']=='USER_VAE100000_CPU_EXPORT_PASSED'
 r=ex['files']['receipt'];s['vae_export']=dict(run='user_vae100k_export_v1',receipt_path='/data/code/ouyanghao/oyh/textop-reproduction-20261024/filtered15000_20261027/vae_export_runs/user_vae100k_export_v1/receipt.json',receipt_sha256=ex['sha256']['receipt'],exit_sha256=ex['sha256']['exit'])
 s['vae_inference']=r['inference']
 for k in ['corpus_receipt_sha256','data_acceptance_sha256','strict_audit_receipt_sha256']:s[k]=r[k]
assert prior['common_lock_free'] and not prior['workers']
code=r'''
import pathlib,json,base64,hashlib,sys,fcntl,shutil,subprocess,os
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
sys.path.insert(0,str(r/'scripts'))
from user_vae100k_completion import stopped_vae100k,accepted_vae100k,read,sha,owned,closed_job
from checkpoint_storage_guard import check
stage=__STAGE__;p=r/'configs'/__NAME__;assert sha(p)==__OLD_SHA__
s=json.loads(base64.b64decode(__SPEC__));lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
v=stopped_vae100k(s['vae_training']) if stage=='quality' else accepted_vae100k(s['vae_training'])
for n,h in s['bound_sources'].items():assert sha(r/n)==h,n
mem=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024;assert mem>=16*1024**3
assert shutil.disk_usage(r).free>=3*1024**3
capacity={}
if stage=='quality':
 assert sha(r/'scripts/validate_user_vae100k.py')==s['entrypoint_sha256'] and sha(r/'scripts/supervise_user_vae100k_quality.py')==s['supervisor_sha256']
 s['quality_authorized']=True
elif stage=='export':
 assert sha(r/'scripts/export_user_vae100k.py')==s['executor_sha256'] and sha(r/'scripts/supervise_user_vae100k_export.py')==s['supervisor_sha256']
 parent=r/'filtered15000_20261027/user_vae100k_inference';assert not parent.exists() and not parent.is_symlink();owned(parent.parent,r)
 needed=max(s['minimum_disk_free_bytes'],s['reserve_bytes']+s['log_allowance_bytes'])+s['export_upper_bound_bytes'];assert shutil.disk_usage(r).free>=needed
 parent.mkdir(mode=0o700);s.update(export_authorized=True,export_allowed_root=str(r),export_dir=str(parent));capacity=dict(path=str(parent),required_bytes=needed,free_bytes=shutil.disk_usage(parent).free)
else:
 es=read(r/'configs/user-vae100k-export-spec.json');w=s['vae_export'];er=read(w['receipt_path'])
 assert er['status']=='USER_VAE100000_CPU_EXPORT_PASSED' and er['output_writers_closed'] and er['exact_inference_reload']
 assert er['source_sha256']==es['executor_sha256'] and er['spec_sha256']==sha(r/'configs/user-vae100k-export-spec.json')
 proof=closed_job('filtered15000_vae_export_'+w['run'],w['receipt_path'],w['receipt_sha256'],es['supervisor_sha256']);assert proof['exit_sha256']==w['exit_sha256']
 assert er['vae_training_receipt_sha256']==v['receipt_sha256'] and er['full_state_checkpoint']==v['checkpoint']
 assert er['inference']==s['vae_inference'] and sha(er['inference']['path'])==er['inference']['sha256'] and er['trained_model_sha256']==v['receipt']['trained_model_sha256']
 assert sha(r/'scripts/run_user_vae100k_ldm.py')==s['entrypoint_sha256'] and sha(r/'scripts/supervise_user_vae100k_ldm.py')==s['supervisor_sha256']
 for k in ['corpus_receipt_sha256','data_acceptance_sha256','strict_audit_receipt_sha256']:assert s[k]==er[k]==v['spec'][k]
 needed=s['checkpoint_policy']['reserve_bytes']+s['checkpoint_policy']['log_allowance_bytes']+4*s['checkpoint_policy']['checkpoint_upper_bound_bytes']+s['vae_cache_upper_bound_bytes']
 allowed=pathlib.Path('/home/ouyanghao');owned(allowed,allowed);parent=allowed/'textop-user-vae100k-ldm-checkpoints-20261028'
 assert not parent.exists() and not parent.is_symlink();assert shutil.disk_usage(allowed).free>=needed,dict(free=shutil.disk_usage(allowed).free,required=needed)
 parent.mkdir(mode=0o700);dest=parent/'checkpoints';dest.mkdir(mode=0o700)
 capacity=check(dest,allowed,s['checkpoint_policy'],remaining_saves=4);capacity.update(total_required_including_cache_bytes=needed,separate_vae_cache_upper_bound_bytes=s['vae_cache_upper_bound_bytes'])
 s.update(witness_authorized=True,train_authorized=False,witnesses=[],checkpoint_allowed_root=str(allowed),checkpoint_dir=str(dest))
if stage!='export':
 free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()};assert free[1]>20480
archive=r/'receipts'/('user_vae100k_'+stage+'_spec_before_enable.json')
with archive.open('xb') as f:f.write(p.read_bytes());f.flush();os.fsync(f.fileno())
raw=(json.dumps(s,indent=2)+'\n').encode();temp=p.with_name(p.name+'.enable.partial')
with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
temp.replace(p)
result=dict(status='USER_VAE100K_'+stage.upper()+'_ENABLED_NOT_DISPATCHED',capacity=capacity,new_spec_sha256=sha(p),new_spec_base64=base64.b64encode(raw).decode(),new_gpu_contexts=0,new_model_loads=0)
with (r/'receipts'/('user_vae100k_'+stage+'_activation.json')).open('x') as f:json.dump(result,f,indent=2)
lock.close();print(json.dumps(result))
'''
for k,v in [('__STAGE__',a.stage),('__NAME__',name),('__OLD_SHA__',hashlib.sha256(old).hexdigest()),('__SPEC__',base64.b64encode(json.dumps(s).encode()).decode())]:code=code.replace(k,repr(v))
ast.parse(code);result=json.loads(recorded_remote(code,O/('activate_'+a.stage+'_v1_client.json'),timeout=120))
raw=base64.b64decode(result['new_spec_base64']);assert hashlib.sha256(raw).hexdigest()==result['new_spec_sha256'];sp.write_bytes(raw)
(O/('activate_'+a.stage+'_v1.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
d=pathlib.Path('D:/具身智能/TextOp复现_20261024/configs');d.mkdir(parents=True,exist_ok=True);shutil.copy2(sp,d/name)
print(json.dumps({k:v for k,v in result.items() if k!='new_spec_base64'}))

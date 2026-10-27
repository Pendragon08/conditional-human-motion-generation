"""Future real VAE200k -> CPU export -> owned-capacity LDM witnesses.

Enables one stage without dispatch. Until actual completed upstream receipts
exist this tool refuses; prepared sources and desired counts cannot activate it.
"""
import argparse,ast,base64,hashlib,json,pathlib,shutil
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['export','witnesses'],required=True);a=p.parse_args()
prior_run='filtered15000_vae200k_train_v1' if a.stage=='export' else 'filtered15000_vae200k_export_v1'
j=json.loads((B/'results/filtered15000_handoff_20261027'/prior_run/'progress.json').read_text(encoding='utf-8'))
assert j['kind']==('vae' if a.stage=='export' else 'vae_export')
assert j['closed_execution_verified'] and j['common_lock_free'] and not j['workers']
rec=j['files']['receipt']
assert rec['status']==('FILTERED15000_VAE_200000_TRAINING_FINISHED' if a.stage=='export' else 'FILTERED15000_REAL_VAE200000_CPU_EXPORT_PASSED')
name='filtered15000-vae-export-spec.json' if a.stage=='export' else 'filtered15000-ldm-spec.json'
sp=B/'configs'/name;old=sp.read_bytes();s=json.loads(old)
assert not s.get('export_authorized',False) and not s.get('witness_authorized',False) and not s.get('train_authorized',False)
root='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
if a.stage=='export':
 assert rec['completed_primitive_attempts']==200000 and rec['vae_quality_gate_passed']
 s['vae_training']=dict(run=prior_run,receipt_path=root+'/filtered15000_20261027/vae_runs/'+prior_run+'/receipt.json',receipt_sha256=j['sha256']['receipt'],exit_sha256=j['sha256']['exit'])
else:
 exs=json.loads((B/'configs/filtered15000-vae-export-spec.json').read_bytes())
 assert rec['exact_inference_reload'] and rec['new_gpu_contexts']==0
 s['vae_training']=exs['vae_training'];s['vae_export']=dict(run=prior_run,
  receipt_path=root+'/filtered15000_20261027/vae_export_runs/'+prior_run+'/receipt.json',receipt_sha256=j['sha256']['receipt'],exit_sha256=j['sha256']['exit'])
 s['vae_inference']=rec['inference']
 for k in ['corpus_receipt_sha256','data_acceptance_sha256','strict_audit_receipt_sha256']:s[k]=rec[k]
code=r'''
import pathlib,json,hashlib,base64,os,sys,fcntl,shutil,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
stage=__STAGE__;p=r/'configs'/__NAME__;assert hashlib.sha256(p.read_bytes()).hexdigest()==__OLD_SHA__
s=json.loads(base64.b64decode(__TEMPLATE__));sys.path.insert(0,str(r/'scripts'))
from filtered15000_vae_completion import completed_vae,closed_job,read,sha,owned
from checkpoint_storage_guard import check
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
v=completed_vae(s['vae_training'])
for n,h in s['bound_sources'].items():assert sha(r/n)==h,n
mem=[int(x.split()[1])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0];assert mem>=16*1024**3
if stage=='export':
 assert not s['export_authorized'] and sha(r/'scripts/export_filtered15000_vae.py')==s['executor_sha256']
 assert sha(r/'scripts/supervise_filtered15000_vae_export.py')==s['supervisor_sha256']
 allowed=r;parent=r/'filtered15000_20261027/vae_inference';assert not parent.exists() and not parent.is_symlink()
 assert shutil.disk_usage(r).free>=max(s['minimum_disk_free_bytes'],s['reserve_bytes']+s['log_allowance_bytes'])+s['export_upper_bound_bytes']
 owned(parent.parent,r);parent.mkdir(mode=0o700)
 fs=subprocess.check_output(['findmnt','-T',str(parent),'-n','-o','FSTYPE'],text=True,timeout=15).strip();assert fs in ['ext4','xfs','btrfs','zfs']
 s.update(export_authorized=True,export_allowed_root=str(allowed),export_dir=str(parent))
 capacity=dict(destination=str(parent),free_bytes=shutil.disk_usage(parent).free,filesystem=fs,export_upper_bound_bytes=s['export_upper_bound_bytes'],capacity_scope='CPU export only; LDM storage not accepted')
 status='ACTUAL_VAE200K_CPU_EXPORT_ENABLED_NOT_DISPATCHED'
else:
 assert not s['witness_authorized'] and not s['train_authorized'] and not s['witnesses']
 es=read(r/'configs/filtered15000-vae-export-spec.json');w=s['vae_export'];er=read(w['receipt_path'])
 assert es['export_authorized'] and er['source_sha256']==es['executor_sha256'] and er['spec_sha256']==sha(r/'configs/filtered15000-vae-export-spec.json')
 proof=closed_job('filtered15000_vae_export_'+w['run'],w['receipt_path'],w['receipt_sha256'],es['supervisor_sha256']);assert proof['exit_sha256']==w['exit_sha256']
 assert er['status']=='FILTERED15000_REAL_VAE200000_CPU_EXPORT_PASSED' and er['output_writers_closed'] and er['exact_inference_reload']
 assert er['inference']==s['vae_inference'] and er['vae_training_receipt_sha256']==v['receipt_sha256']
 assert sha(er['inference']['path'])==er['inference']['sha256'] and er['trained_model_sha256']==v['receipt']['trained_model_sha256']
 assert sha(r/'scripts/run_filtered15000_ldm.py')==s['entrypoint_sha256'] and sha(r/'scripts/supervise_filtered15000_ldm.py')==s['supervisor_sha256']
 for k in ['corpus_receipt_sha256','data_acceptance_sha256','strict_audit_receipt_sha256']:assert s[k]==er[k]==v['spec'][k]
 needed=s['checkpoint_policy']['reserve_bytes']+s['checkpoint_policy']['log_allowance_bytes']+4*s['checkpoint_policy']['checkpoint_upper_bound_bytes']+s['vae_cache_upper_bound_bytes']
 # These are owned task destinations on two previously established disks;
 # choose only an actually sufficient persistent location without deleting data.
 candidates=[(r,r/'filtered15000_20261027/ldm'),(pathlib.Path('/home/ouyanghao'),pathlib.Path('/home/ouyanghao/textop-filtered15000-ldm-20261027'))]
 selected=None;observed=[]
 for allowed,parent in candidates:
  owned(allowed,allowed);observed.append(dict(allowed_root=str(allowed),free_bytes=shutil.disk_usage(allowed).free,required_bytes=needed))
  if not parent.exists() and not parent.is_symlink() and shutil.disk_usage(allowed).free>=needed:
   selected=(allowed,parent);break
 assert selected,'Preserve all states: no actual persistent capacity for LDM; measurements='+json.dumps(observed)
 allowed,parent=selected;owned(parent.parent,allowed);parent.mkdir(mode=0o700);dest=parent/'checkpoints';dest.mkdir(mode=0o700)
 capacity=check(dest,allowed,s['checkpoint_policy'],remaining_saves=4)
 capacity['separate_vae_cache_upper_bound_bytes']=s['vae_cache_upper_bound_bytes'];capacity['total_required_including_cache_bytes']=needed
 free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()};assert free[1]>20480
 s.update(witness_authorized=True,train_authorized=False,witnesses=[],checkpoint_allowed_root=str(allowed),checkpoint_dir=str(dest))
 status='ACTUAL_FILTERED_LDM_WITNESSES_ENABLED_NOT_DISPATCHED'
archive=r/'receipts'/('filtered15000_'+stage+'_spec_before_enable.json')
with archive.open('xb') as f:f.write(p.read_bytes())
raw=(json.dumps(s,indent=2)+'\n').encode();tmp=p.with_name(p.name+'.enable.partial')
with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
tmp.replace(p)
result=dict(status=status,capacity=capacity,new_spec_sha256=sha(p),new_spec_base64=base64.b64encode(raw).decode(),new_model_loads=0,new_gpu_contexts=0,new_training_steps=0)
with (r/'receipts'/('filtered15000_'+stage+'_activation.json')).open('x') as f:json.dump(result,f,indent=2)
lock.close();print(json.dumps(result))
'''
for key,value in [('__STAGE__',a.stage),('__NAME__',name),('__OLD_SHA__',hashlib.sha256(old).hexdigest()),('__TEMPLATE__',base64.b64encode(json.dumps(s).encode()).decode())]:code=code.replace(key,repr(value))
ast.parse(code)
folder=B/'results/filtered15000_handoff_tools_20261027'
result=json.loads(recorded_remote(code,folder/('activate_ldm_'+a.stage+'_v1_client.json'),timeout=90))
raw=base64.b64decode(result['new_spec_base64']);assert hashlib.sha256(raw).hexdigest()==result['new_spec_sha256']
sp.write_bytes(raw)
(folder/('ldm_'+a.stage+'_activation.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
d=pathlib.Path('D:/具身智能/TextOp复现_20261024/configs');d.mkdir(parents=True,exist_ok=True);shutil.copy2(sp,d/name)
print(json.dumps({k:v for k,v in result.items() if k!='new_spec_base64'}))

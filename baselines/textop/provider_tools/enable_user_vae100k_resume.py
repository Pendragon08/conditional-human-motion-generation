"""Enable only the final300 formal updates after real independent recovery witnesses."""
import argparse,ast,base64,hashlib,json,pathlib
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1];O=B/'results/vae100k_user_handoff_20261028';p=argparse.ArgumentParser();p.add_argument('--document-summary',required=True);a=p.parse_args()
dp=pathlib.Path(a.document_summary);doc=json.loads(dp.read_bytes());sp=B/'configs/user-vae100k-resume-spec.json';old=sp.read_bytes();s=json.loads(old)
assert s['witness_authorized'] and not s['train_authorized'] and not s['witnesses']
assert doc['status']=='USER_VAE100K_RESUME_FRESH_DOCUMENT_VERIFIED' and doc['run']=='r100_document_v1' and doc['actual_exit_code']==0 and doc['findings']==[]
assert doc['resume_checkpoint_sha256']==s['resume_checkpoint']['sha256'] and doc['entrypoint_sha256']==s['entrypoint_sha256']
backup_path=O/'r997/backup.json'
if backup_path.exists():
 backup=json.loads(backup_path.read_bytes());assert backup['status']=='REAL99700_EMERGENCY_PRIVATE_BACKUP_VERIFIED' and backup['checkpoint']==s['resume_checkpoint'] and backup['D_sha256']==s['resume_checkpoint']['sha256']
 backup_status='VERIFIED'
else:
 intent=json.loads((O/'r997/intent.json').read_bytes());assert intent['checkpoint']==s['resume_checkpoint']
 backup_status='PENDING_BYTE_COPY_NOT_ACCEPTED'

for run in ['r100_author_v1','r100_document_v1']:
 j=json.loads((O/run/'progress.json').read_bytes());r=j['files']['receipt']
 assert j['kind']=='resume' and j['closed_execution_verified'] and not j['workers'] and j['common_lock_free']
 assert r['status']=='USER_VAE100K_RESUME_WITNESS_PASSED' and r['source_sha256']==s['entrypoint_sha256'] and r['resume_checkpoint']==s['resume_checkpoint'] and r['new_optimizer_updates']==4
 assert r['finite_gradients'] and r['exact_model_adam_rng_nextbatch_restore'] and r['pending_validation_without_updates'] and not r['diagnostic_state_reused_for_training'] and not r['persistent_checkpoint_created']
 s['witnesses'].append(dict(run=run,receipt_sha256=j['sha256']['receipt'],exit_sha256=j['sha256']['exit']))
s.update(train_authorized=True,fresh_document_summary_sha256=hashlib.sha256(dp.read_bytes()).hexdigest(),private_backup_status=backup_status,private_backup_receipt_sha256=hashlib.sha256(backup_path.read_bytes()).hexdigest() if backup_path.exists() else None,private_backup_intent_sha256=hashlib.sha256((O/'r997/intent.json').read_bytes()).hexdigest())
raw=(json.dumps(s,ensure_ascii=False,indent=2)+'\n').encode()
code=r'''
import base64,fcntl,json,os,pathlib,sys,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013;sys.path.insert(0,str(r/'scripts'))
from user_vae100k_resume_protocol import failed_source,read,sha,closed_job
from checkpoint_storage_guard import check
p=r/'configs/user-vae100k-resume-spec.json';assert sha(p)==__OLD__;raw=base64.b64decode(__RAW__);s=json.loads(raw)
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);v=failed_source(s)
assert not pathlib.Path(s['target_checkpoint_path']).exists()
for w in s['witnesses']:
 rp=r/'vae100k_user_handoff_20261028'/w['run']/'receipt.json';wr=read(rp);proof=closed_job('user_vae100k_resume_'+w['run'],rp,w['receipt_sha256'],s['supervisor_sha256'])
 assert proof['exit_sha256']==w['exit_sha256'] and wr['status']=='USER_VAE100K_RESUME_WITNESS_PASSED' and wr['source_sha256']==s['entrypoint_sha256'] and wr['resume_checkpoint']==s['resume_checkpoint']
 assert wr['finite_gradients'] and wr['exact_model_adam_rng_nextbatch_restore'] and wr['pending_validation_without_updates'] and not wr['diagnostic_state_reused_for_training']
capacity=check(pathlib.Path(v['spec']['checkpoint_dir']),pathlib.Path(v['spec']['checkpoint_allowed_root']),v['spec']['checkpoint_policy'],remaining_saves=2)
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()};assert free[1]>20480
with (r/'receipts/user100k_resume_spec_before_train.json').open('xb') as f:f.write(p.read_bytes());f.flush();os.fsync(f.fileno())
temp=p.with_name(p.name+'.train.partial')
with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
temp.replace(p);print(json.dumps(dict(status='REAL99700_RECOVERY_ACCEPTED_300_UPDATES_ENABLED_NOT_DISPATCHED',spec_sha256=sha(p),capacity=capacity,new_gpu_contexts=0,new_model_loads=0)))
'''.replace('__OLD__',repr(hashlib.sha256(old).hexdigest())).replace('__RAW__',repr(base64.b64encode(raw).decode()))
ast.parse(code);z=json.loads(recorded_remote(code,O/'renable_client.json',timeout=120));sp.write_bytes(raw);(O/'renable.json').write_text(json.dumps(z,indent=2),encoding='utf-8');print(json.dumps(z))

"""Back up the real small private bundle to D; enable/dispatch bounded acceptance."""
import pathlib,json,hashlib,shutil,base64,sys,ast
from backup_filter_recovery_checkpoint import recorded_remote,recorded_client
from filter_recovery_windows_paths import guard_local
B=pathlib.Path(__file__).resolve().parents[1];D=pathlib.Path('D:/具身智能/TextOp复现_20261024');ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
run='filtered15000_recovery_v1';parent=B/'results/filtered15000_handoff_20261027';j=json.loads((parent/run/'progress.json').read_text(encoding='utf-8'))
assert j['kind']=='recovery' and j['closed_execution_verified'] and j['common_lock_free'] and not j['workers']
rec=j['files']['receipt'];item=rec['private_archive'];assert rec['status']=='FILTERED15000_CORPUS_RECOVERY_PREPARED' and rec['output_writers_closed']
assert item['path']==ROOT+'/filtered15000_20261027/data_runs/'+run+'/corpus_recovery.private.tar.gz' and 0<item['bytes']<128*1024**2
out=parent/'private_recovery_backup_v1';out.mkdir(parents=True,exist_ok=False)
guard_local(D,D,directory=True);folder=D/'datasets/filtered15000_private_20261027/recovery';folder.mkdir(parents=True,exist_ok=True);guard_local(D,folder,directory=True)
target=folder/'corpus_recovery_v1.private.tar.gz';partial=target.with_name(target.name+'.partial')
guard_local(D,target,must_absent=True);guard_local(D,partial,must_absent=True)
assert shutil.disk_usage(D).free>=2*1024**3+item['bytes']
cmd=['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','-i','C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32:'+item['path'],str(partial)]
client=recorded_client('',out/'scp_client.json',timeout=900,command=cmd)
assert client['actual_client_exit_code']==0 and client['client_reaped'] and not client['timed_out'] and client['error'] is None
guard_local(D,partial);assert partial.stat().st_size==item['bytes']
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
assert sha(partial)==item['sha256'];guard_local(D,target,must_absent=True);partial.replace(target);guard_local(D,target)
backup=dict(status='CORPUS_RECOVERY_SERVER_D_SHA_VERIFIED',server_archive=item['path'],archive_bytes=item['bytes'],archive_sha256=item['sha256'],
 D_archive=str(target),D_sha256=sha(target),D_owner_path_verified=True,D_reserve_bytes_after=shutil.disk_usage(D).free,
 recovery_receipt_sha256=j['sha256']['receipt'],recovery_supervisor_sha256=j['sha256']['exit'],actual_scp_exit_code=0,
 scp_client_evidence_sha256=sha(out/'scp_client.json'),formal_training_started=False)
assert backup['D_reserve_bytes_after']>=2*1024**3
(out/'backup.json').write_text(json.dumps(backup,ensure_ascii=False,indent=2),encoding='utf-8')
sp=B/'configs/filtered15000-recovery-spec.json';old=sp.read_bytes();s=json.loads(old);assert s['prepare_authorized'] and not s['acceptance_authorized']
s['acceptance_authorized']=True;data=(json.dumps(s,indent=2)+'\n').encode();expected=hashlib.sha256(old).hexdigest()
code=r'''
import pathlib,json,base64,hashlib,os,sys,fcntl
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
backup=json.loads(base64.b64decode(__BACKUP_JSON__));p=r/'receipts/filtered15000_corpus_recovery_private_backup.json'
sys.path.insert(0,str(r/'scripts'));from filtered15000_corpus_acceptance import sha,read,owned
rp=r/'filtered15000_20261027/data_runs/filtered15000_recovery_v1/receipt.json';rec=read(rp)
assert sha(rp)==backup['recovery_receipt_sha256'] and rec['status']=='FILTERED15000_CORPUS_RECOVERY_PREPARED'
arc=owned(pathlib.Path(backup['server_archive']),r);assert sha(arc)==backup['archive_sha256'] and arc.stat().st_size==backup['archive_bytes']
assert backup['archive_sha256']==rec['private_archive']['sha256'] and backup['actual_scp_exit_code']==0 and backup['D_reserve_bytes_after']>=2*1024**3
sp=r/'configs/filtered15000-recovery-spec.json';assert sha(sp)==__OLD_SPEC__
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
with p.open('x') as f:json.dump(backup,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
archive=r/'receipts/filtered15000_recovery_spec_before_acceptance.json'
with archive.open('xb') as f:f.write(sp.read_bytes())
raw=base64.b64decode(__NEW_BYTES__);temp=sp.with_name(sp.name+'.accept_enable.partial')
with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
temp.replace(sp);lock.close()
print(json.dumps(dict(status='PRIVATE_BACKUP_RECORDED_ACCEPTANCE_ENABLED_NOT_STARTED',new_spec_sha256=sha(sp),private_backup_receipt_sha256=sha(p))))
'''.replace('__BACKUP_JSON__',repr(base64.b64encode(json.dumps(backup,ensure_ascii=False).encode()).decode())).replace('__OLD_SPEC__',repr(expected)).replace('__NEW_BYTES__',repr(base64.b64encode(data).decode()))
ast.parse(code)
activation=json.loads(recorded_remote(code,out/'record_private_backup_client.json',timeout=90));sp.write_bytes(data)
(out/'acceptance_activation.json').write_text(json.dumps(activation,indent=2),encoding='utf-8')
dispatch=recorded_client('',out/'acceptance_dispatch_client.json',timeout=120,command=[sys.executable,str(B/'provider_tools/launch_filtered15000_handoff.py'),'--stage','accept_data','--run','filtered15000_acceptance_v1'])
assert dispatch['actual_client_exit_code']==0 and dispatch['client_reaped'] and not dispatch['timed_out'] and dispatch['error'] is None
print(json.dumps(dict(status='PRIVATE_RECOVERY_BACKED_UP_ACCEPTANCE_DISPATCHED_NOT_YET_VERIFIED',archive_bytes=item['bytes'],D_private_archive=str(target),new_generator_started=False)))

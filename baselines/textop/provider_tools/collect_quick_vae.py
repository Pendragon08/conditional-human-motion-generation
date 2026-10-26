"""Collect actual quick-VAE evidence; optionally back up stable real snapshots privately."""
import argparse,hashlib,json,pathlib,shutil,subprocess
from session import remote
B=pathlib.Path(__file__).resolve().parents[1];D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--backup-stable-checkpoints',action='store_true');a=p.parse_args();assert a.run.isidentifier() and a.run.isascii()
script=r'''
import pathlib,json,hashlib,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');run=RUN_LITERAL
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rp=r/'quick_generator_20261026/vae_runs'/run/'receipt.json';rec=json.loads(rp.read_text());ep=r/'receipts'/('quick_vae_'+run+'_exit.json')
result={'receipt':rec,'receipt_sha256':sha(rp),'exit':None,'checkpoints_stable':[]}
if ep.exists():
 ex=json.loads(ep.read_text());assert ex['evaluation_receipt_sha256']==sha(rp)
 for f,k in [(r/'receipts'/('quick_vae_'+run+'_process_exit.json'),'process_exit_sha256'),(r/'logs'/('quick_vae_'+run+'_resources.jsonl'),'resources_sha256')]:assert sha(f)==ex[k]
 result['exit']=ex;result['exit_sha256']=sha(ep)
for ck in rec.get('checkpoints',[])+([rec['emergency_checkpoint']] if 'emergency_checkpoint' in rec else []):
 f=pathlib.Path(ck['path']);assert f.parent==r/'quick_generator_20261026/vae/checkpoints'
 assert f.stat().st_size==ck['bytes'] and sha(f)==ck['sha256']
 if time.time()-f.stat().st_mtime>30:result['checkpoints_stable'].append(ck)
print(json.dumps(result))
'''.replace('RUN_LITERAL',repr(a.run))
j=json.loads(remote(script,timeout=80));out=B/'results/quick_vae_20261026'/a.run;out.mkdir(parents=True,exist_ok=True)
(out/'receipt.json').write_text(json.dumps(j['receipt'],indent=2),encoding='utf-8')
if j['exit'] is not None:(out/'exit.json').write_text(json.dumps(j['exit'],indent=2),encoding='utf-8')
backups=[]
if a.backup_stable_checkpoints:
 import msvcrt
 guard=(D/'quick_vae_checkpoint_backup.lock').open('a+b')
 if guard.seek(0,2)==0:guard.write(b'0');guard.flush()
 guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
 for ck in j['checkpoints_stable']:
  folder=D/'checkpoints/private/quick_vae_20261026';folder.mkdir(parents=True,exist_ok=True);target=folder/pathlib.PurePosixPath(ck['path']).name
  if not target.exists():
   assert shutil.disk_usage(folder).free>ck['bytes']+1024**3
   partial=target.with_suffix('.pth.partial');assert not partial.exists(),'Preserve interrupted copy; inspect before recovery'
   z=subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','-i','C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32:'+ck['path'],str(partial)],capture_output=True,text=True,timeout=300)
   assert z.returncode==0,z.stderr
   assert partial.stat().st_size==ck['bytes'] and hashlib.sha256(partial.read_bytes()).hexdigest()==ck['sha256'];partial.replace(target)
  assert target.stat().st_size==ck['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==ck['sha256']
  backups.append({**ck,'private_server_and_local_sha_verified':True,'publicly_uploaded':False})
 (out/'checkpoint_backup.json').write_text(json.dumps({'checkpoints':backups,'deleted_files':0},indent=2),encoding='utf-8')
local=D/'results/quick_vae_20261026'/a.run;local.mkdir(parents=True,exist_ok=True)
for f in out.iterdir():
 if f.is_file():shutil.copy2(f,local/f.name);assert f.read_bytes()==(local/f.name).read_bytes()
r=j['receipt'];print(json.dumps({'run':a.run,'status':r['status'],'mode':r['mode'],'completed_primitive_attempts':r['completed_primitive_attempts'],
 'optimizer_updates':r['optimizer_updates'],'skips':r['skipped_updates'],'exit_code':j['exit']['child_exit_code'] if j['exit'] else None,
 'backed_up_checkpoints':len(backups),'full_paper_training':False}))

"""One user-authorized real pause; exact frozen sources, process and CPU proof."""
import argparse, fcntl, gzip, hashlib, json, os, pathlib, shutil, time
from closed_boundary_interrupt import identity, interrupt_at_guard, require_identity
from checkpoint_storage_guard import check

ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
sp=ROOT/'configs/generator8278-pause-spec.json';s=json.loads(sp.read_text())
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert s['schema']=='textop.generator8278_user_pause.v1' and os.getuid()==1013
assert set(os.sched_getaffinity(0)) <= {0,1,2,3}
assert s['entrypoint_sha256']==sha(__file__)
for n,h in s['frozen_sources'].items():assert sha(ROOT/n)==h,n
guard=(ROOT/'receipts/generator8278_pause_control.lock').open('a+')
fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
expected=s['process_identity'];require_identity(identity(expected['pid']),expected)
assert expected['argv']==[str(ROOT/'env/bin/python'),str(ROOT/'scripts/run_generator8278_vae.py'),'--mode','train','--run',s['run']]
assert identity(expected['pid'])['state'] not in ('T','t')
spec=json.loads((ROOT/'configs/generator8278-vae-spec.json').read_text())
rp=ROOT/'generator8278_20261027/vae_runs'/s['run']/'receipt.json';r=json.loads(rp.read_text())
assert r['status']=='RUNNING_GENERATOR8278_VAE_TRAINING' and r['completed_primitive_attempts']>=100
assert r['spec_sha256']==s['frozen_sources']['configs/generator8278-vae-spec.json']
dest=pathlib.Path(spec['checkpoint_dir']);assert not list(dest.glob('emergency_*.pth'))
budget=check(dest,ROOT,spec['checkpoint_policy'],remaining_saves=1)
assert str(pathlib.Path(shutil.which('findmnt')).resolve())==s['findmnt_executable']
out=ROOT/'generator_pause_20261027'/s['handoff_name']
assert not out.exists(),'An existing/pending control must be inspected, never repeated'
if not a.execute:
 print(json.dumps({'status':'PRODUCTION_PAUSE_PREFLIGHT_ONLY','training_signals':0,'worker':expected,'budget':budget}));raise SystemExit(0)
assert s['pause_authorized'] is True
for w in s['cpu_witnesses']:
 f=ROOT/'pause_control_fixtures'/w['run']/'receipt.json';j=json.loads(f.read_text())
 assert sha(f)==w['sha256'] and j['status']=='CPU_BOUNDARY_INTERRUPT_FIXTURE_PASSED'
 assert j['actual_fixture_exit']==41 and j['training_signals']==0 and j['gpu_runs']==0
 assert j['source_sha256']['closed_boundary_interrupt.py']==s['frozen_sources']['scripts/closed_boundary_interrupt.py']
assert len(s['cpu_witnesses'])==2
out.mkdir(parents=True,exist_ok=False)
record={'status':'WAITING_CLOSED_BATCH_BOUNDARY','reason':s['user_request'],'run':s['run'],
 'original_source_files_changed':False,'spec_sha256':sha(sp),'preflight_worker':expected,'preflight_budget':budget,
 'training_completed':False,'filtered_generator_training':False}
def save():
 f=out/'receipt.partial';f.write_text(json.dumps(record,indent=2),encoding='utf-8');f.replace(out/'receipt.json')
save()
try:
 record['control']=interrupt_at_guard(expected,['findmnt','-T',str(dest),'-n','-o','FSTYPE'],s['findmnt_executable'],out/'signals.jsonl',600,600)
 r=json.loads(rp.read_text());assert r['status']=='FAILED_GENERATOR8278_VAE' and r['error']=='KeyboardInterrupt()'
 ck=r['emergency_checkpoint'];f=pathlib.Path(ck['path'])
 assert f.parent==dest and f.name=='emergency_'+str(ck['manager_step'])+'.pth'
 assert ck['manager_step']==r['completed_primitive_attempts'] and ck['actual_optimizer_updates']==r['optimizer_updates']
 assert ck['manager_step']%4==0 and f.stat().st_size==ck['bytes']<=spec['checkpoint_policy']['checkpoint_upper_bound_bytes']
 assert sha(f)==ck['sha256']
 record.update(checkpoint=ck,original_receipt=r,original_receipt_sha256=sha(rp),status='CHECKPOINT_WRITTEN_WAITING_SUPERVISOR_EXIT');save()
 exits={k:ROOT/'receipts'/('generator8278_vae_'+s['run']+'_'+k+'.json') for k in ['exit','process_exit']}
 exits['outer_exit']=ROOT/'generator8278_20261027/drivers'/(s['run']+'_outer_exit.json')
 end=time.monotonic()+120
 while time.monotonic()<end and not all(f.exists() for f in exits.values()):time.sleep(.5)
 assert all(f.exists() for f in exits.values()),'Original supervisor/driver exits not yet all present'
 proof={k:json.loads(f.read_text()) for k,f in exits.items()}
 assert proof['process_exit']['child_reaped'] and proof['process_exit']['child_pid']==expected['pid']
 lock=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
 metrics=rp.parent/'metrics.jsonl.gz';lines=0
 with gzip.open(metrics,'rt',encoding='utf-8') as fh:
  for line in fh:json.loads(line);lines+=1
 record.update(status='USER_PAUSED_GENERATOR_CHECKPOINT_PRIVATE_BACKUP_PENDING',original_exits=proof,
  exit_hashes={k:sha(f) for k,f in exits.items()},gpu_common_lock_released=True,
  metrics_gzip_fully_decoded=True,metric_records=lines,original_false_closed_writer_field_retained=True,
  actual_worker_exited=True,private_backup_completed=False,checkpoint_cpu_restore_audit_completed=False)
 save();print(json.dumps(record))
except BaseException as e:
 record.update(status='PAUSE_HANDOFF_NEEDS_INSPECTION',error=repr(e));save();raise

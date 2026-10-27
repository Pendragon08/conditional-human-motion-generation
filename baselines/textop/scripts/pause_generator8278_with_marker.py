"""Explicit owned-temp-path save request; never sends process signals."""
import argparse,fcntl,gzip,hashlib,json,os,pathlib,re,time
from closed_boundary_marker import request_boundary_save
from closed_boundary_interrupt import identity,require_identity
from checkpoint_storage_guard import check
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
sp=ROOT/'configs/generator8278-marker-pause-spec.json';s=json.loads(sp.read_text());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert s['schema']=='textop.generator8278_owned_marker_pause.v1' and os.getuid()==1013
assert set(os.sched_getaffinity(0))<={0,1,2,3} and sha(__file__)==s['entrypoint_sha256']
for n,h in s['frozen_sources'].items():assert sha(ROOT/n)==h,n
guard=(ROOT/'receipts/generator8278_pause_control.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
expected=s['process_identity'];require_identity(identity(expected['pid']),expected)
assert expected['argv']==[str(ROOT/'env/bin/python'),str(ROOT/'scripts/run_generator8278_vae.py'),'--mode','train','--run',s['run']]
assert identity(expected['pid'])['state'] not in ('T','t')
spec=json.loads((ROOT/'configs/generator8278-vae-spec.json').read_text());dest=pathlib.Path(spec['checkpoint_dir'])
rp=ROOT/'generator8278_20261027/vae_runs'/s['run']/'receipt.json';r=json.loads(rp.read_text())
assert r['status']=='RUNNING_GENERATOR8278_VAE_TRAINING' and r['completed_primitive_attempts']>=100
assert r['spec_sha256']==s['frozen_sources']['configs/generator8278-vae-spec.json']
assert not list(dest.glob('emergency_*.pth')) and not list(dest.glob('*.partial'))
budget=check(dest,ROOT,spec['checkpoint_policy'],remaining_saves=1)
out=ROOT/'generator_pause_20261027'/s['handoff_name'];assert not out.exists()
assert not (rp.parent/'receipt.partial').is_dir()
if not a.execute:
 print(json.dumps({'status':'MARKER_PAUSE_PREFLIGHT_ONLY','training_signals':0,'worker':expected,'budget':budget}));raise SystemExit(0)
assert s['pause_authorized'] is True and len(s['cpu_witnesses'])==2
for w in s['cpu_witnesses']:
 f=ROOT/'pause_marker_fixtures'/w['run']/'receipt.json';j=json.loads(f.read_text());assert sha(f)==w['sha256']
 assert j['status']=='CPU_CLOSED_BOUNDARY_MARKER_PASSED' and all(j['checks'].values()) and j['gpu_calls']==0
 assert j['source_sha256']['closed_boundary_marker.py']==s['frozen_sources']['scripts/closed_boundary_marker.py']
out.mkdir(parents=True,exist_ok=False)
record={'status':'WAITING_USER_REQUESTED_BOUNDARY_SAVE','reason':s['user_request'],'run':s['run'],'spec_sha256':sha(sp),
 'source_files_changed':False,'training_signals':0,'training_completed':False,'prefilter_data':True,'preflight_budget':budget}
def save():
 p=out/'handoff.partial';p.write_text(json.dumps(record,indent=2),encoding='utf-8');p.replace(out/'receipt.json')
save()
try:
 record['control']=request_boundary_save(expected,rp.parent,out,600)
 files=list(dest.glob('emergency_*.pth'));assert len(files)==1
 f=files[0];step=int(re.fullmatch(r'emergency_(\d+)\.pth',f.name).group(1));assert step%4==0
 assert f.stat().st_size<=spec['checkpoint_policy']['checkpoint_upper_bound_bytes']
 ck={'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f),'manager_step':step,'actual_optimizer_updates':None}
 record.update(checkpoint=ck,status='CHECKPOINT_WRITTEN_WAITING_REAL_EXITS');save()
 paths={k:ROOT/'receipts'/('generator8278_vae_'+s['run']+'_'+k+'.json') for k in ['exit','process_exit']}
 paths['outer_exit']=ROOT/'generator8278_20261027/drivers'/(s['run']+'_outer_exit.json')
 end=time.monotonic()+120
 while time.monotonic()<end and not all(p.exists() for p in paths.values()):time.sleep(.5)
 proofs={k:json.loads(p.read_text()) for k,p in paths.items()}
 assert proofs['process_exit']['child_reaped'] and proofs['process_exit']['child_pid']==expected['pid']
 assert proofs['process_exit']['child_exit_code']==1 and not proofs['process_exit']['timed_out']
 log=ROOT/'logs'/('generator8278_vae_'+s['run']+'.log');text=log.read_text()
 first=text.split('Traceback (most recent call last):',1)[1].split('During handling of the above exception',1)[0]
 assert 'IsADirectoryError' in first and str(rp.parent/'receipt.partial') in first
 assert 'run_generator8278_vae.py", line '+str(s['normal_boundary_save_lineno'])+', in <module>' in first
 lock=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
 lines=0
 with gzip.open(rp.parent/'metrics.jsonl.gz','rt',encoding='utf-8') as stream:
  for line in stream:json.loads(line);lines+=1
 record.update(status='USER_PAUSED_GENERATOR_CHECKPOINT_PRIVATE_BACKUP_PENDING',original_receipt=json.loads(rp.read_text()),
  original_receipt_stale_running_retained=True,original_exits=proofs,exit_hashes={k:sha(p) for k,p in paths.items()},
  log_sha256=sha(log),controlled_exception='IsADirectoryError at normal closed-boundary receipt write; existing emergency save completed',
  gpu_common_lock_released=True,metrics_gzip_fully_decoded=True,metric_records=lines,
  actual_worker_exited=True,private_backup_completed=False,checkpoint_cpu_restore_audit_completed=False)
 save();print(json.dumps(record))
except BaseException as e:
 record.update(status='PAUSE_HANDOFF_NEEDS_INSPECTION',error=repr(e));save();raise

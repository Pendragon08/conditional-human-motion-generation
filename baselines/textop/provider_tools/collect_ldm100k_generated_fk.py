"""Light live FK status; one terminal immutable manifest/NPZ byte audit."""
import argparse
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--run', choices=['user_ldm100k_generated_fk_v1', 'user_ldm100k_generated_fk_v2'], default='user_ldm100k_generated_fk_v2')
p.add_argument('--evidence', required=True)
a = p.parse_args()
assert a.evidence.isidentifier() and a.evidence.isascii()
run = a.run
out = B / 'results/ldm100k_user_handoff_20261028' / run
code = r'''
import json,pathlib,os,fcntl,sys,time,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'))
from ldm100k_verified_input import owned,read,sha
run=RUN;prefix='ldm100k_generated_fk_'+run
request=owned(r/'receipts'/(prefix+'_request.json'));s=read(request);parent=owned(s['output_root'],pathlib.Path('/home/ouyanghao'))
log=parent/(run+'_supervision');files={};hashes={}
for key,p in dict(receipt=parent/run/'receipt.json',exit=log/'exit.json',process=log/'process_exit.json',outer=r/'receipts'/(prefix+'_outer_exit.json')).items():
 if p.exists():files[key]=read(p);hashes[key]=sha(p)
workers=[]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  argv=(proc/'cmdline').read_bytes().decode().split('\0')
  if any(str(r/'scripts'/n) in argv for n in ['export_ldm100k_generated_fk.py','supervise_ldm100k_generated_fk.py']):
   st=(proc/'stat').read_text().rsplit(')',1)[1].split();workers.append(dict(pid=int(proc.name),start_ticks=st[19],state=st[0],argv=argv))
 except (FileNotFoundError,ProcessLookupError):pass
lock=(r/'receipts/full_library_capacity.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
except BlockingIOError:free=False
finally:lock.close()
logs={}
for p in [log/'child.log',parent/(run+'_outer.log')]:
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2000));logs[p.name]=f.read().decode(errors='replace')
closed=all(k in files for k in ['receipt','exit','process','outer']) and files['outer']['actual_returncode']==0 and files['exit']['supervision_passed'] and files['process']['child_exit_code']==0 and not workers and free
audit=None
if closed:
 rec=files['receipt'];assert rec['status']=='USER_LDM100K_GENERATED_ISAAC_FK_FINISHED' and rec['output_writers_closed']
 assert rec['source_sha256']==s['executor_sha256'] and files['exit']['source_sha256']==s['supervisor_sha256']
 assert hashes['receipt']==files['exit']['receipt_sha256'] and sha(request)==files['exit']['request_sha256']
 assert hashes['process']==files['exit']['process_exit_sha256']
 assert sha(log/'resources.jsonl')==files['exit']['resource_sha256']
 samples=[json.loads(x) for x in (log/'resources.jsonl').read_text().splitlines()]
 assert samples[-1]['phase']=='after_exit' and samples[-1]['owned_group_processes']==[]
 m=rec['private_manifest'];mp=owned(m['path'],pathlib.Path('/home/ouyanghao'));assert sha(mp)==m['sha256'] and mp.stat().st_size==m['bytes']
 summary=read(s['summary_receipt']);ip=summary['private_artifacts']['physical_inputs.private.json'];assert sha(ip['path'])==ip['sha256']
 expected={(x['segment_id'],x['seed'],x['mode']):x for x in read(ip['path'])};seen=set();count=frames=total=omissions=0
 for line in mp.read_text().splitlines():
  x=json.loads(line);key=x['segment_id'],x['seed'],x['mode'];assert key in expected and key not in seen;seen.add(key)
  original=expected[key];assert all(x[k]==v for k,v in original.items())
  if x['status']=='NO_TRANSITION_SINGLE_FRAME_RETAINED':assert x['frames']==1 and not x['physical_failure_claimed'];omissions+=1;continue
  assert x['status']=='REAL_GENERATED_ISAAC_FK_EXPORTED' and x['no_joint_clamp_applied'] and not x['dynamics_executed']
  p=owned(x['file'],pathlib.Path('/home/ouyanghao'));assert p.parent==parent/run and p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
  assert x['robot_joint_names']==rec['joint_names'] and x['robot_body_names']==rec['body_names']
  count+=1;frames+=x['frames'];total+=x['bytes']
 assert seen==set(expected) and count==rec['exported_trials'] and frames==rec['exported_frames'] and total==rec['private_fk_npz_bytes'] and omissions==rec['no_transition_trials']
 audit=dict(manifest_and_all_private_npz_bytes_sha_verified=True,all_declared_seed_mode_trials_covered=True,exported_trials=count,exported_frames=frames,no_transition_trials=omissions,model_loads=0,new_gpu_contexts=0,dynamics_executed=False,physical_quality_accepted=False)
print(json.dumps(dict(run=run,observed_unix=time.time(),files=files,sha256=hashes,workers=workers,common_lock_free=free,closed_execution_verified=bool(closed),artifact_audit=audit,logs=logs,disks={str(p):shutil.disk_usage(p).free for p in [r,parent]})))
'''.replace('RUN', repr(run))
j = json.loads(recorded_remote(code, out / (a.evidence + '_client.json'), timeout=180))
(out / (a.evidence + '.json')).write_text(json.dumps(j, indent=2), encoding='utf-8')
(out / 'progress.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
print(json.dumps(j))

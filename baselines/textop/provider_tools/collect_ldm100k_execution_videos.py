"""Read actual media batch; verify completed MP4 bytes and bounded child exits."""
import argparse
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--evidence', required=True)
a = p.parse_args()
assert a.evidence.isascii() and a.evidence.isidentifier()
run = 'user_ldm100k_execution_videos_v1'
out = B / 'results/ldm100k_user_handoff_20261028' / run
code = r'''
import json,pathlib,sys,os,fcntl,time,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));from ldm100k_verified_input import read,sha,owned
run='user_ldm100k_execution_videos_v1';prefix='ldm100k_execution_videos_'+run
sp=r/'receipts'/(prefix+'_request.json');s=read(sp);parent=owned(s['output_root'],pathlib.Path('/home/ouyanghao'))
files={};hashes={}
for key,p in dict(receipt=parent/'batch_receipt.json',outer=r/'receipts'/(prefix+'_outer_exit.json')).items():
 if p.exists():files[key]=read(p);hashes[key]=sha(p)
workers=[]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  argv=(proc/'cmdline').read_bytes().decode().split('\0')
  if any(str(r/'scripts'/n) in argv for n in ['render_ldm100k_execution_video.py','supervise_ldm100k_execution_videos.py']):
   st=(proc/'stat').read_text().rsplit(')',1)[1].split();workers.append(dict(pid=int(proc.name),start_ticks=st[19],state=st[0],argv=argv))
 except (FileNotFoundError,ProcessLookupError):pass
lock=(r/'receipts/full_library_capacity.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
except BlockingIOError:free=False
finally:lock.close()
logs={}
for p in [parent/'outer.log']+sorted(parent.glob('*.log'))[-2:]:
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2500));logs[p.name]=f.read().decode(errors='replace')
rec=files.get('receipt',{});verified=[]
for item in rec.get('completed_clips',[]):
 clip=item['clip'];rp=owned(parent/clip/'receipt.json',pathlib.Path('/home/ouyanghao'));proof=read(parent/(clip+'_process_exit.json'))
 assert sha(rp)==item['receipt_sha256'] and proof['supervision_passed'] and proof['child_exit_code']==0 and proof['child_reaped']
 rr=read(rp);video=owned(item['video_path'],pathlib.Path('/home/ouyanghao'))
 assert video.stat().st_size==item['video_bytes'] and sha(video)==item['video_sha256']
 assert rr['nvidia_libraries_loaded']==[] and 'llvmpipe' in rr['gl_renderer'].lower() and rr['frames_encoded']==rr['frames_decoded']==item['frames']
 assert rr['new_physics_steps']==rr['new_optimizer_updates']==rr['sdk_runs']==0
 samples=[json.loads(x) for x in (parent/(clip+'_resources.jsonl')).read_text().splitlines()]
 assert samples[-1]['phase']=='after_exit' and not samples[-1]['owned_group_processes']
 verified.append(item)
closed=rec.get('status')=='USER_LDM100K_RECORDED_EXECUTION_VIDEOS_FINISHED' and files.get('outer',{}).get('actual_returncode')==0 and not workers and free
if closed:assert rec['output_writers_closed'] and len(verified)==rec['declared_clips']==len(s['selected_trial_indices'])
print(json.dumps(dict(run=run,observed_unix=time.time(),files=files,sha256=hashes,workers=workers,common_lock_free=free,closed_execution_verified=bool(closed),verified_media=verified,logs=logs,home_free_bytes=shutil.disk_usage(parent).free)))
'''
j = json.loads(recorded_remote(code, out / (a.evidence + '_client.json'), timeout=180))
(out / (a.evidence + '.json')).write_text(json.dumps(j, indent=2), encoding='utf-8')
(out / 'progress.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
print(json.dumps(j))

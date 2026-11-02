"""Create the signed real-media scope and one CPU/Mesa batch, never replay physics."""
import ast
import base64
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
O = B / 'results/ldm100k_user_handoff_20261028'
RUN = 'user_ldm100k_execution_videos_v1'
PHYSICS = 'user_ldm100k_generated_physics_v1'
H = '/home/ouyanghao/textop-user100k-evaluation-20261029'
R = '/data/code/ouyanghao/oyh/textop-reproduction-20261024'
j = json.loads((O / PHYSICS / 'progress.json').read_bytes())
assert j['closed_execution_verified'] and not j['workers'] and j['common_lock_free']
assert j['artifact_audit']['all1880_strict_trials_and_terminal_poses_verified']
s = json.loads((B / 'configs/execution-video-spec.json').read_bytes())
sources = {}
for name in ['scripts/render_ldm100k_execution_video.py', 'scripts/supervise_ldm100k_execution_videos.py']:
    raw = (B / name).read_bytes()
    ast.parse(raw, feature_version=(3, 10))
    sources[name] = dict(raw=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
s.update(run=RUN, output_root=H + '/' + RUN, maximum_frames=201, maximum_video_bytes=8 * 1024**2,
    quality_label='ACTUAL_RECORDED_LDM100K_GENERATED_FROZEN15K_PHYSICS',
    renderer_sha256=sources['scripts/render_ldm100k_execution_video.py']['sha256'],
    supervisor_sha256=sources['scripts/supervise_ldm100k_execution_videos.py']['sha256'],
    prior_visual_spec_sha256=hashlib.sha256((B / 'configs/execution-video-spec.json').read_bytes()).hexdigest(),
    physics_binding={
        'receipt': dict(path=H + '/' + PHYSICS + '/receipt.json', sha256=j['sha256']['receipt']),
        'exit': dict(path=H + '/' + PHYSICS + '_supervision/exit.json', sha256=j['sha256']['exit']),
        'process': dict(path=H + '/' + PHYSICS + '_supervision/process_exit.json', sha256=j['sha256']['process']),
        'outer': dict(path=R + '/receipts/ldm100k_generated_physics_' + PHYSICS + '_outer_exit.json', sha256=j['sha256']['outer'])})
out = O / RUN
out.mkdir(exist_ok=False)
code = r'''
import os,sys,json,pathlib,base64,hashlib,subprocess,shutil,fcntl
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));os.sched_setaffinity(0,{0,1,2,3})
from ldm100k_verified_input import read,sha,owned
from contact_filter_resources import sample_resources
run=RUN;s=REQUEST;prefix='ldm100k_execution_videos_'+run
assert not any((r/'receipts'/(prefix+x)).exists() for x in ['_driver.py','_request.json','_driver.json','_outer_exit.json'])
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
resources=sample_resources(r,None,2*1024**3,2*1024**3+128*1024**2)
for item in s['physics_binding'].values():assert sha(item['path'])==item['sha256']
physics=read(s['physics_binding']['receipt']['path']);assert physics['output_writers_closed'] and physics['finished_trials']==1880
index=pathlib.Path(s['physics_binding']['receipt']['path']).parent/'executed_traces.private.json'
assert sha(index)==physics['private_executed_trace_index_sha256'];rows=read(index)
key=min((x['segment_id'],x['seed']) for x in rows)
first=[i for i,x in enumerate(rows) if (x['segment_id'],x['seed'])==key];assert len(first)==2
failed=[i for i,x in enumerate(rows) if x['physical_failure']];assert len(failed)==7
s['selected_trial_indices']=first+[i for i in failed if i not in first]
parent=pathlib.Path(s['output_root']);assert parent.parent==pathlib.Path('/home/ouyanghao/textop-user100k-evaluation-20261029')
assert not parent.exists() and shutil.disk_usage(parent.parent).free>2*1024**3+128*1024**2+64*1024**2
parent.mkdir(mode=0o700)
for name,item in SOURCES.items():
 p=r/name;assert not p.exists();raw=base64.b64decode(item['raw']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 assert sha(p)==item['sha256']
request=r/'receipts'/(prefix+'_request.json')
with request.open('x') as f:json.dump(s,f,indent=2);f.flush();os.fsync(f.fileno())
lock.close();cmd=['python3',str(r/'scripts/supervise_ldm100k_execution_videos.py'),'--request',str(request)]
outer=r/'receipts'/(prefix+'_outer_exit.json');driver=r/'receipts'/(prefix+'_driver.py')
text="import subprocess,pathlib,json,os\np=subprocess.run("+repr(cmd)+")\nwith pathlib.Path("+repr(str(outer))+").open('x') as f:json.dump({'actual_returncode':p.returncode},f);f.flush();os.fsync(f.fileno())\nraise SystemExit(p.returncode)\n"
with driver.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
with (parent/'outer.log').open('xb') as f:q=subprocess.Popen(['python3',str(driver)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
result=dict(status='ACTUAL_EXECUTION_VIDEO_BATCH_DISPATCHED_NOT_ACCEPTED',run=run,driver_pid=q.pid,request_sha256=sha(request),selected_trial_indices=s['selected_trial_indices'],declared_clips=len(s['selected_trial_indices']),preflight_resources=resources)
with (r/'receipts'/(prefix+'_driver.json')).open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(result))
'''.replace('RUN', repr(RUN)).replace('REQUEST', repr(s)).replace('SOURCES', repr(sources))
ast.parse(code, feature_version=(3, 8))
result = json.loads(recorded_remote(code, out / 'dispatch_client.json', timeout=180))
(out / 'dispatch.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))

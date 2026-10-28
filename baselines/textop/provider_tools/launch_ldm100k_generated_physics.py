"""Serial inference-only strict replay of the actual completed generated FK."""
import ast
import base64
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
RUN = 'user_ldm100k_generated_physics_v1'
FK = 'user_ldm100k_generated_fk_v2'
HOME = '/home/ouyanghao/textop-user100k-evaluation-20261029'
R = '/data/code/ouyanghao/oyh/textop-reproduction-20261024'
j = json.loads((B / 'results/ldm100k_user_handoff_20261028' / FK / 'progress.json').read_bytes())
assert j['closed_execution_verified'] and not j['workers'] and j['common_lock_free']
assert j['artifact_audit']['all_declared_seed_mode_trials_covered']
sources = {}
for name in ['scripts/evaluate_ldm100k_generated_physics.py', 'scripts/supervise_ldm100k_generated_physics.py']:
    raw = (B / name).read_bytes()
    ast.parse(raw, feature_version=(3, 10))
    sources[name] = dict(raw=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
bound = {name: hashlib.sha256((B / name).read_bytes()).hexdigest() for name in [
    'scripts/evaluate_teacher15000_strict.py', 'scripts/teacher15000_strict_protocol.py', 'scripts/direct_strict_trace.py',
    'scripts/strict_tracking_coverage.py', 'scripts/paper_filter_config.py', 'scripts/filter_training_runtime.py',
    'scripts/validated_contact_profile.py', 'scripts/bounded_headless_close.py', 'scripts/run_tracker.sh',
    'configs/tracker-env-spec.json', 'configs/teacher15000-direct-strict-spec.json', 'configs/filter-training-plan.json']}
request = dict(run=RUN, fk_receipt=HOME + '/' + FK + '/receipt.json', fk_receipt_sha256=j['sha256']['receipt'],
    fk_closure={HOME + '/' + FK + '_supervision/' + key + '.json': j['sha256'][value]
                for key, value in [('process_exit', 'process'), ('exit', 'exit')]},
    executor_sha256=sources['scripts/evaluate_ldm100k_generated_physics.py']['sha256'],
    supervisor_sha256=sources['scripts/supervise_ldm100k_generated_physics.py']['sha256'], bound_sources=bound,
    policy_checkpoint_sha256='66e87c9471d2860f1b43a18a41e938c12647c93212f1fe44c8be1f1b27febc88',
    output_root=HOME, reserve_bytes=2 * 1024**3, log_allowance_bytes=512 * 1024**2,
    output_upper_bytes=195580 * 36 * 4 + 96 * 1024**2, training_updates_authorized=0)
request['fk_closure'][R + '/receipts/ldm100k_generated_fk_' + FK + '_outer_exit.json'] = j['sha256']['outer']
out = B / 'results/ldm100k_user_handoff_20261028' / RUN
out.mkdir(exist_ok=False)
code = r'''
import json,base64,os,pathlib,hashlib,fcntl,sys,subprocess,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));os.sched_setaffinity(0,{0,1,2,3})
from ldm100k_verified_input import read,sha,owned
from contact_filter_resources import sample_resources
run=RUN;s=REQUEST;prefix='ldm100k_generated_physics_'+run
parent=owned(s['output_root'],pathlib.Path('/home/ouyanghao'))
assert not (parent/run).exists() and not (parent/(run+'_supervision')).exists()
assert not any((r/'receipts'/(prefix+x)).exists() for x in ['_driver.py','_driver.json','_request.json','_outer_exit.json'])
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
resources=sample_resources(r,None,16*1024**3,2*1024**3+128*1024**2)
gpu=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
assert gpu>=20*1024**3
assert shutil.disk_usage(parent).free>=s['reserve_bytes']+s['log_allowance_bytes']+s['output_upper_bytes']
assert sha(owned(s['fk_receipt'],pathlib.Path('/home/ouyanghao')))==s['fk_receipt_sha256']
for path,h in s['fk_closure'].items():
 assert sha(path)==h
 proof=read(path);assert proof.get('actual_returncode',0)==0 and proof.get('child_exit_code',0)==0 and proof.get('supervision_passed',True)
for name,h in s['bound_sources'].items():assert sha(owned(r/name))==h,name
for name,item in SOURCES.items():
 p=r/name;assert not p.exists()
 raw=base64.b64decode(item['raw']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 assert sha(p)==item['sha256']
request=r/'receipts'/(prefix+'_request.json')
with request.open('x') as f:json.dump(s,f,indent=2);f.flush();os.fsync(f.fileno())
lock.close()
cmd=['python3',str(r/'scripts/supervise_ldm100k_generated_physics.py'),'--request',str(request)]
outer=r/'receipts'/(prefix+'_outer_exit.json');driver=r/'receipts'/(prefix+'_driver.py')
text="import subprocess,pathlib,json,os\np=subprocess.run("+repr(cmd)+")\nwith pathlib.Path("+repr(str(outer))+").open('x') as f:json.dump({'actual_returncode':p.returncode},f);f.flush();os.fsync(f.fileno())\nraise SystemExit(p.returncode)\n"
with driver.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
with (parent/(run+'_outer.log')).open('xb') as f:q=subprocess.Popen(['python3',str(driver)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
result=dict(status='GENERATED_PHYSICS_DISPATCHED_NOT_ACCEPTED',run=run,driver_pid=q.pid,request=s,request_sha256=sha(request),preflight_resources=resources,gpu1_free_bytes=gpu,home_free_bytes=shutil.disk_usage(parent).free)
with (r/'receipts'/(prefix+'_driver.json')).open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(result))
'''.replace('RUN', repr(RUN)).replace('REQUEST', repr(request)).replace('SOURCES', repr(sources))
ast.parse(code, feature_version=(3, 8))
j = json.loads(recorded_remote(code, out / 'dispatch_client.json', timeout=180))
(out / 'dispatch.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
print(json.dumps(j))

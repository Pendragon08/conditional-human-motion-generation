"""One recorded-state position metric extraction, from actually closed sources."""
import ast
import base64
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
O = B / 'results/ldm100k_user_handoff_20261028'
RUN = 'user_ldm100k_tracking_fidelity_v1'
HOME = '/home/ouyanghao/textop-user100k-evaluation-20261029'
ROOT = '/data/code/ouyanghao/oyh/textop-reproduction-20261024'
physics = 'user_ldm100k_generated_physics_v1'
j = json.loads((O / physics / 'progress.json').read_bytes())
media = json.loads((O / 'user_ldm100k_execution_videos_v1/progress.json').read_bytes())
assert j['closed_execution_verified'] and media['closed_execution_verified']
assert j['artifact_audit']['all1880_strict_trials_and_terminal_poses_verified']
fk = json.loads((O / 'user_ldm100k_generated_fk_v2/progress.json').read_bytes())
sources = {}
for name in ['scripts/measure_ldm100k_tracking_fidelity.py', 'scripts/supervise_ldm100k_tracking_fidelity.py']:
    raw = (B / name).read_bytes()
    ast.parse(raw, feature_version=(3, 10))
    sources[name] = dict(raw=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
binding = {key: dict(path=HOME + '/' + physics + '_supervision/' + file, sha256=j['sha256'][key])
           for key, file in [('exit', 'exit.json'), ('process', 'process_exit.json')]}
binding['receipt'] = dict(path=HOME + '/' + physics + '/receipt.json', sha256=j['sha256']['receipt'])
binding['outer'] = dict(path=ROOT + '/receipts/ldm100k_generated_physics_' + physics + '_outer_exit.json', sha256=j['sha256']['outer'])
s = dict(run=RUN, physics_binding=binding, output_root=HOME,
    fk_receipt=dict(path=HOME + '/user_ldm100k_generated_fk_v2/receipt.json', sha256=fk['sha256']['receipt']),
    executor_sha256=sources['scripts/measure_ldm100k_tracking_fidelity.py']['sha256'],
    supervisor_sha256=sources['scripts/supervise_ldm100k_tracking_fidelity.py']['sha256'],
    bound_sources={name: hashlib.sha256((B / name).read_bytes()).hexdigest() for name in [
        'scripts/export_tracker_references.py', 'scripts/bounded_headless_close.py', 'scripts/run_tracker.sh', 'configs/tracker-env-spec.json']},
    reserve_bytes=2 * 1024**3, log_allowance_bytes=128 * 1024**2, output_upper_bytes=96 * 1024**2,
    source_scope='all1880 existing recorded physics traces; no newly generated data',
    full_coverage_required_for_success=True, root_relative_success_threshold_m=.3,
    new_neural_model_loads_authorized=0, new_policy_rollouts_authorized=0, new_physics_steps_authorized=0)
out = O / RUN
out.mkdir(exist_ok=False)
code = r'''
import json,pathlib,hashlib,base64,fcntl,os,sys,shutil,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));os.sched_setaffinity(0,{0,1,2,3})
from ldm100k_verified_input import read,sha,owned
from contact_filter_resources import sample_resources
run=RUN;s=SPEC;prefix='ldm100k_tracking_fidelity_'+run
parent=owned(s['output_root'],pathlib.Path('/home/ouyanghao'))
assert not (parent/run).exists() and not (parent/(run+'_supervision')).exists()
assert not any((r/'receipts'/(prefix+x)).exists() for x in ['_driver.py','_driver.json','_outer_exit.json','_request.json'])
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
resources=sample_resources(r,None,16*1024**3,2*1024**3+128*1024**2)
assert shutil.disk_usage(parent).free>s['reserve_bytes']+s['log_allowance_bytes']+s['output_upper_bytes']
gpu=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
assert gpu>=20*1024**3
for item in s['physics_binding'].values():assert sha(item['path'])==item['sha256']
for name,h in s['bound_sources'].items():assert sha(owned(r/name))==h
for name,item in SOURCES.items():
 p=r/name;assert not p.exists();raw=base64.b64decode(item['raw']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 assert sha(p)==item['sha256']
sp=r/'receipts'/(prefix+'_request.json')
with sp.open('x') as f:json.dump(s,f,indent=2);f.flush();os.fsync(f.fileno())
lock.close();cmd=['python3',str(r/'scripts/supervise_ldm100k_tracking_fidelity.py'),'--request',str(sp)]
driver=r/'receipts'/(prefix+'_driver.py');outer=r/'receipts'/(prefix+'_outer_exit.json')
text="import subprocess,pathlib,json,os\np=subprocess.run("+repr(cmd)+")\nwith pathlib.Path("+repr(str(outer))+").open('x') as f:json.dump({'actual_returncode':p.returncode},f);f.flush();os.fsync(f.fileno())\nraise SystemExit(p.returncode)\n"
with driver.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
with (parent/(run+'_outer.log')).open('xb') as f:q=subprocess.Popen(['python3',str(driver)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
result=dict(status='RECORDED_TRACKING_FIDELITY_DISPATCHED_NOT_ACCEPTED',run=run,driver_pid=q.pid,request_sha256=sha(sp),preflight_resources=resources,gpu1_free_bytes=gpu,home_free_bytes=shutil.disk_usage(parent).free)
with (r/'receipts'/(prefix+'_driver.json')).open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(result))
'''.replace('RUN', repr(RUN)).replace('SPEC', repr(s)).replace('SOURCES', repr(sources))
ast.parse(code, feature_version=(3, 8))
result = json.loads(recorded_remote(code, out / 'dispatch_client.json', timeout=180))
(out / 'dispatch.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))

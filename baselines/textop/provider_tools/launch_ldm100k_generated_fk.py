"""Dispatch genuine generated-motion FK only after actual closed CPU audit."""
import ast
import base64
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
RUN = 'user_ldm100k_generated_fk_v2'
S = 'user_ldm100k_diagnostic_summary_v2'
j = json.loads((B / 'results/ldm100k_user_handoff_20261028' / S / 'progress.json').read_bytes())
assert j['closed_execution_verified'] and not j['workers'] and j['common_lock_free']
rec = j['files']['receipt']
assert rec['actual_trials'] == 1880 and rec['crc_float32_finite_shape_data_sha_checks_passed']
source = {}
for name in ['scripts/export_ldm100k_generated_fk.py', 'scripts/supervise_ldm100k_generated_fk.py']:
    raw = (B / name).read_bytes()
    ast.parse(raw, feature_version=(3, 10))
    source[name] = dict(raw=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
bind = {name: hashlib.sha256((B / name).read_bytes()).hexdigest() for name in
        ['scripts/export_tracker_references.py', 'scripts/bounded_headless_close.py', 'configs/tracker-env-spec.json']}
spec = json.loads((B / 'configs/tracker-env-spec.json').read_bytes())
request = dict(run=RUN, summary_run=S, summary_receipt='/home/ouyanghao/textop-user100k-evaluation-20261029/' + S + '/receipt.json',
    summary_receipt_sha256=j['sha256']['receipt'], summary_closure_sha256={
        '_process_exit.json': j['sha256']['process'], '_exit.json': j['sha256']['exit'], '_outer_exit.json': j['sha256']['outer']},
    executor_sha256=source['scripts/export_ldm100k_generated_fk.py']['sha256'],
    supervisor_sha256=source['scripts/supervise_ldm100k_generated_fk.py']['sha256'], bound_sources=bind,
    tracker_spec_sha256=bind['configs/tracker-env-spec.json'], urdf_sha256=spec['robot_urdf_sha256'],
    original_fk_source_sha256=bind['scripts/export_tracker_references.py'], close_helper_sha256=bind['scripts/bounded_headless_close.py'],
    output_root='/home/ouyanghao/textop-user100k-evaluation-20261029', reserve_bytes=2 * 1024**3,
    log_allowance_bytes=128 * 1024**2,
    output_upper_bytes=195580 * 2152 + 1880 * 8192 + 64 * 1024**2,
    physics_steps_authorized=0, training_updates_authorized=0)
out = B / 'results/ldm100k_user_handoff_20261028' / RUN
out.mkdir(exist_ok=False)
code = r'''
import json,base64,os,pathlib,hashlib,fcntl,sys,subprocess,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));os.sched_setaffinity(0,{0,1,2,3})
from ldm100k_verified_input import read,sha,owned
from contact_filter_resources import sample_resources
run=RUN;s=REQUEST;prefix='ldm100k_generated_fk_'+run
parent=owned(s['output_root'],pathlib.Path('/home/ouyanghao'))
assert not (parent/run).exists() and not (parent/(run+'_supervision')).exists()
assert not any((r/'receipts'/(prefix+x)).exists() for x in ['_driver.py','_driver.json','_request.json','_outer_exit.json'])
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
resources=sample_resources(r,None,16*1024**3,2*1024**3+128*1024**2)
gpu=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
assert gpu>=20*1024**3
assert shutil.disk_usage(parent).free>=s['reserve_bytes']+s['log_allowance_bytes']+s['output_upper_bytes']
rp=owned(s['summary_receipt'],pathlib.Path('/home/ouyanghao'));assert sha(rp)==s['summary_receipt_sha256']
summary=read(rp);assert summary['status']=='USER_LDM100K_DIAGNOSTIC32_CPU_SUMMARY_FINISHED'
assert summary['actual_trials']==1880 and summary['actual_pairs']==940
for suffix,digest in s['summary_closure_sha256'].items():assert sha(r/'receipts'/('ldm100k_diagnostic_summary_'+s['summary_run']+suffix))==digest
for name,h in s['bound_sources'].items():assert sha(owned(r/name))==h
s['wrapper_sha256']=sha(owned(r/'scripts/run_tracker.sh'))
old='user_ldm100k_generated_fk_v1';old_log=parent/(old+'_supervision')
assert read(r/'receipts'/('ldm100k_generated_fk_'+old+'_outer_exit.json'))['actual_returncode']==1
proof=read(old_log/'process_exit.json');assert proof['child_reaped'] and proof['child_exit_code']==1
samples=[json.loads(x) for x in (old_log/'resources.jsonl').read_text().splitlines()]
assert samples[-1]['phase']=='after_exit' and samples[-1]['owned_group_processes']==[]
old_rec=read(parent/old/'receipt.json');assert old_rec['exported_trials']==0 and old_rec['physics_steps']==0
assert (parent/old/'fk_manifest.private.jsonl').stat().st_size==0
archive=r/'receipts/ldm100k_fk_failed_v1';archive.mkdir(exist_ok=False)
expected={'scripts/export_ldm100k_generated_fk.py':'a319dbbdcd62a6e4f2ed0399f05b0fa6bb2dcad6302151dc58e1b9278081a19c','scripts/supervise_ldm100k_generated_fk.py':'eab5d74a32eed97695e0165bd4a6f0e188f6bfd23c726c8305a941d8a45ee69d'}
for name,h in expected.items():
 p=owned(r/name);assert sha(p)==h
 with (archive/p.name).open('xb') as f:f.write(p.read_bytes());f.flush();os.fsync(f.fileno())
 assert sha(archive/p.name)==h
for name,item in SOURCES.items():
 p=r/name;assert sha(p)==expected[name]
 raw=base64.b64decode(item['raw']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
 with p.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 assert sha(p)==item['sha256']
request=r/'receipts'/(prefix+'_request.json')
with request.open('x') as f:json.dump(s,f,indent=2);f.flush();os.fsync(f.fileno())
lock.close()
cmd=['python3',str(r/'scripts/supervise_ldm100k_generated_fk.py'),'--request',str(request)]
outer=r/'receipts'/(prefix+'_outer_exit.json');driver=r/'receipts'/(prefix+'_driver.py')
text="import subprocess,pathlib,json,os\np=subprocess.run("+repr(cmd)+")\nwith pathlib.Path("+repr(str(outer))+").open('x') as f:json.dump({'actual_returncode':p.returncode},f);f.flush();os.fsync(f.fileno())\nraise SystemExit(p.returncode)\n"
with driver.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
with (parent/(run+'_outer.log')).open('xb') as f:q=subprocess.Popen(['python3',str(driver)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
result=dict(status='GENERATED_FK_DISPATCHED_NOT_ACCEPTED',run=run,driver_pid=q.pid,request=s,request_sha256=sha(request),preflight_resources=resources,gpu1_free_bytes=gpu,home_free_bytes=shutil.disk_usage(parent).free)
with (r/'receipts'/(prefix+'_driver.json')).open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(result))
'''.replace('RUN', repr(RUN)).replace('REQUEST', repr(request)).replace('SOURCES', repr(source))
ast.parse(code, feature_version=(3, 8))
j = json.loads(recorded_remote(code, out / 'dispatch_client.json', timeout=180))
(out / 'dispatch.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
print(json.dumps(j))

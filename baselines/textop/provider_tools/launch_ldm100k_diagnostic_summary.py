"""Deploy and launch one immutable, bounded CPU analysis of closed generation."""
import ast
import base64
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
RUN = 'user_ldm100k_diagnostic_summary_v2'
out = B / 'results/ldm100k_user_handoff_20261028' / RUN
out.mkdir(exist_ok=False)
names = ['scripts/summarize_ldm100k_diagnostic.py', 'scripts/supervise_ldm100k_diagnostic_summary.py']
source = {}
for name in names:
    raw = (B / name).read_bytes()
    ast.parse(raw, feature_version=(3, 8))
    source[name] = dict(raw=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
j = json.loads((B / 'results/ldm100k_user_handoff_20261028/user_ldm100k_diagnostic32_v1/progress.json').read_bytes())
assert j['closed_execution_verified'] and not j['workers'] and j['common_lock_free']
request = dict(run=RUN, generation_receipt_sha256=j['sha256']['receipt'],
    generation_spec_sha256=j['files']['receipt']['spec_sha256'],
    executor_sha256=source[names[0]]['sha256'], supervisor_sha256=source[names[1]]['sha256'],
    disk_floor_bytes=3 * 1024**3, new_neural_model_loads_authorized=0, new_gpu_contexts_authorized=0,
    output_root='/home/ouyanghao/textop-user100k-evaluation-20261029',
    allowed_root='/home/ouyanghao', metadata_upper_bytes=16 * 1024**2,
    corpus_spec_sha256=hashlib.sha256((B / 'configs/filtered15000-corpus-spec.json').read_bytes()).hexdigest())
raw = (json.dumps(request, indent=2) + '\n').encode()
(out / 'request.json').write_bytes(raw)
code = r'''
import base64,fcntl,hashlib,json,os,pathlib,subprocess,sys
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
from ldm100k_verified_input import owned,sha,read
from contact_filter_resources import sample_resources
run=RUN;prefix='ldm100k_diagnostic_summary_'+run;request=REQUEST
paths=[r/'receipts'/(prefix+x) for x in ['_driver.py','_driver.json','_exit.json','_process_exit.json','_outer_exit.json']]
assert not any(p.exists() for p in paths) and not (r/'ldm100k_user_handoff_20261028'/run).exists()
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
resources=sample_resources(r,None,16*1024**3,request['disk_floor_bytes'])
old='ldm100k_diagnostic_summary_user_ldm100k_diagnostic_summary_v1'
assert read(r/'receipts'/(old+'_outer_exit.json'))['actual_returncode']==1
assert read(r/'receipts'/(old+'_process_exit.json'))['child_reaped']
archive=r/'receipts/ldm100k_summary_failed_v1';archive.mkdir(exist_ok=False)
expected={'scripts/summarize_ldm100k_diagnostic.py':'e8a87bf333f18caa71cedb2f457c92b58ea47533f258a25b3ffcca1bd5103db2','scripts/supervise_ldm100k_diagnostic_summary.py':'1dc1d8c3ac8de9b53868198ff20008b456c2d01f5850c26b63e4957fe9dea3b2'}
for name,h in expected.items():
 p=owned(r/name);assert sha(p)==h
 with (archive/p.name).open('xb') as f:f.write(p.read_bytes());f.flush();os.fsync(f.fileno())
 assert sha(archive/p.name)==h
parent=pathlib.Path(request['output_root']);assert parent.parent==pathlib.Path('/home/ouyanghao')
parent.mkdir(mode=0o700,exist_ok=True);owned(parent,pathlib.Path('/home/ouyanghao'))
assert __import__('shutil').disk_usage(parent).free>=2*1024**3+request['metadata_upper_bytes']+128*1024**2
for name,item in SOURCES.items():
 p=r/name;assert sha(p)==expected[name]
 raw=base64.b64decode(item['raw']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
 with p.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 assert sha(owned(p))==item['sha256']
rp=r/'receipts'/(prefix+'_request.json')
with rp.open('xb') as f:f.write(base64.b64decode(REQUEST_RAW));f.flush();os.fsync(f.fileno())
assert read(rp)==request
lock.close()
cmd=['python3',str(r/'scripts/supervise_ldm100k_diagnostic_summary.py'),'--request',str(rp)]
outer=r/'receipts'/(prefix+'_outer_exit.json');driver=r/'receipts'/(prefix+'_driver.py')
text="import subprocess,json,pathlib,os\nq=subprocess.run("+repr(cmd)+")\nwith pathlib.Path("+repr(str(outer))+").open('x') as f:json.dump({'actual_returncode':q.returncode},f);f.flush();os.fsync(f.fileno())\nraise SystemExit(q.returncode)\n"
with driver.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
with (r/'logs'/(prefix+'_outer.log')).open('xb') as f:q=subprocess.Popen(['python3',str(driver)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
result=dict(status='CPU_SUMMARY_DISPATCHED_NOT_ACCEPTED',run=run,driver_pid=q.pid,request_sha256=sha(rp),preflight_resources=resources)
with paths[1].open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(result))
'''.replace('RUN', repr(RUN)).replace('REQUEST_RAW', repr(base64.b64encode(raw).decode())).replace('REQUEST', repr(request)).replace('SOURCES', repr(source))
ast.parse(code, feature_version=(3, 8))
result = json.loads(recorded_remote(code, out / 'dispatch_client.json', timeout=180))
(out / 'dispatch.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))

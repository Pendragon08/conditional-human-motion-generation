"""Deploy only inactive new files; verify CPU evidence logic without touching SDK jobs."""
import ast,base64,hashlib,json,pathlib,shutil,subprocess,sys
from session import remote
B=pathlib.Path(__file__).resolve().parents[1]
R=B/'results/filter_contact_protocol_20261026';R.mkdir(exist_ok=True)
files=['scripts/apply_tracking_filter_v3.py','scripts/evaluate_strict_filter_contact.py','scripts/finalize_contact_filter.py','scripts/supervise_strict_contact_replay.py','configs/strict-contact-replay-spec.json','tests/test_filter_contact_v3.py']
bound=json.loads((B/'configs/formal-contact-activation.json').read_text(encoding='utf-8'))['deployed_files']
assert not set(files).intersection(bound)
payload={n:{'bytes':base64.b64encode((B/n).read_bytes()).decode(),'sha256':hashlib.sha256((B/n).read_bytes()).hexdigest()} for n in files}
script=r"""
import ast,base64,hashlib,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
payload=PAYLOAD;bound=BOUND
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(ROOT/n)==s for n,s in bound.items()),'Frozen formal runtime changed before operation'
for n,v in payload.items():
 p=ROOT/n;assert p.parent.resolve().is_relative_to(ROOT) if hasattr(p.parent.resolve(),'is_relative_to') else str(p.parent.resolve()).startswith(str(ROOT)+'/')
 assert p.parent.stat().st_uid==1013 and not p.is_symlink()
 if p.exists():assert sha(p)==v['sha256'],'Refuse replacing different existing candidate'
 else:
  data=base64.b64decode(v['bytes']);assert hashlib.sha256(data).hexdigest()==v['sha256']
  with p.open('xb') as f:f.write(data)
 assert sha(p)==v['sha256']
 if p.suffix=='.py':ast.parse(p.read_text())
run=ROOT/'tracker_validation/filter_contact_protocol_cpu_server_v1';assert not run.exists()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
command=[sys.executable,str(ROOT/'tests/test_filter_contact_v3.py'),'--output',str(run)]
r=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=90)
assert r.returncode==0,(r.stdout,r.stderr)
check=json.loads((run/'receipt.json').read_text())
sys.path.insert(0,str(ROOT/'scripts'))
from apply_tracking_filter_v3 import require_completed_training,contact_signature
active=ROOT/'tracker_validation/filter_policy_paper69_solid_floor_20261026_v1/receipt.json'
activebytes=active.read_bytes();training=json.loads(activebytes)
try:require_completed_training(training)
except AssertionError as ex:rejection=str(ex)
else:raise AssertionError('Running actual policy was wrongly accepted')
assert training['status']=='RUNNING_FORMAL_FILTER_TRAINING'
profile_sha=contact_signature(training['contact_profile'])
# Disabled supervisor must exit before looking for policy/SDK/GPU resources.
disabled=subprocess.run([sys.executable,str(ROOT/'scripts/supervise_strict_contact_replay.py'),'--run','disabled_gate_cpu_only','--checkpoint','unused','--training-receipt','unused','--training-exit-receipt','unused'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=20)
assert disabled.returncode!=0 and 'Future exporter disabled; no SDK dispatched' in disabled.stderr
assert not (ROOT/'tracker_validation/disabled_gate_cpu_only').exists()
assert all(sha(ROOT/n)==s for n,s in bound.items()),'Frozen formal runtime changed during operation'
result={'status':'CONTACT_FILTER_V3_CPU_SERVER_CHECKS_PASSED','python_version':sys.version,'cpu_affinity':sorted(os.sched_getaffinity(0)),'deployed':{n:v['sha256'] for n,v in payload.items()},'frozen_runtime_files_verified':len(bound),'synthetic_check_receipt':check,'test_exit_code':r.returncode,'test_stdout':r.stdout,'test_stderr':r.stderr,'actual_running_policy_rejected':True,'active_policy_status':training['status'],'active_policy_completed_iterations':training['completed_learning_iterations'],'active_policy_updates':training['optimizer_updates'],'active_policy_receipt_sha256':hashlib.sha256(activebytes).hexdigest(),'actual_contact_profile_sha256':profile_sha,'rejection':rejection,'disabled_supervisor_exit_code':disabled.returncode,'disabled_supervisor_stderr':disabled.stderr,'sdk_dispatches':0,'physics_calls':0,'optimizer_updates':0,'physical_filter_executed':False,'dataset_accepted':False,'time_unix':time.time(),'real_exporter_validation_pending':True,'all_fixtures_preserved':True}
(run/'server_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
""".replace('PAYLOAD',repr(payload)).replace('BOUND',repr(bound))
result=json.loads(remote(script,timeout=140));(R/'server_cpu_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
local=B.parents[3]/'textop_reproduction_20261024/filter_contact_v3_author_v3/receipt.json'
assert local.exists();shutil.copy2(local,R/'local_cpu_v3.json')
print(json.dumps({k:result[k] for k in ['status','frozen_runtime_files_verified','actual_running_policy_rejected','active_policy_completed_iterations','active_policy_updates','sdk_dispatches','real_exporter_validation_pending']}))

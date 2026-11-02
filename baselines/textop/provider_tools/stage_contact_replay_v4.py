"""Stage a disabled bounded-validation route; CPU checks only while training owns SDK."""
import base64
import hashlib
import json
import pathlib
import shutil
import subprocess
from session import SSH

B = pathlib.Path(__file__).resolve().parents[1]
R = B / 'results/contact_replay_v4_20261026'
R.mkdir(exist_ok=True)
names = ['scripts/contact_replay_modes.py', 'scripts/evaluate_contact_filter_v4.py',
         'scripts/supervise_contact_filter_v4.py', 'configs/contact-filter-v4-spec.json',
         'tests/test_contact_replay_modes.py']
payload = {n: {'base64': base64.b64encode((B / n).read_bytes()).decode(),
               'sha256': hashlib.sha256((B / n).read_bytes()).hexdigest()} for n in names}
bound = json.loads((B / 'configs/formal-contact-activation.json').read_text(encoding='utf-8'))['deployed_files']
older = json.loads((B / 'results/contact_supervision_20261026/candidate_runtime_bytes_v2.json').read_text(encoding='utf-8'))['files']
older = {n: v['sha256'] for n, v in older.items()}
assert not set(names).intersection(bound) and not set(names).intersection(older)
local = B.parents[3] / 'textop_reproduction_20261024/contact_replay_v4_cpu_author_v1/receipt.json'
assert local.exists()
script = r'''
import ast,base64,fcntl,hashlib,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
payload=PAYLOAD;bound=BOUND;older=OLDER
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(ROOT/n)==s for n,s in bound.items())
assert all(sha(ROOT/n)==s for n,s in older.items())
for n,v in payload.items():
 p=ROOT/n;assert p.parent.stat().st_uid==1013 and not p.is_symlink()
 if p.exists():assert sha(p)==v['sha256'],'Preserve different existing source'
 else:
  with p.open('xb') as f:f.write(base64.b64decode(v['base64']))
 assert sha(p)==v['sha256']
 if p.suffix=='.py':ast.parse(p.read_text())
spec=json.loads((ROOT/'configs/contact-filter-v4-spec.json').read_text())
assert spec['formal_evaluation_enabled'] is False and spec['real_sdk_validation_completed'] is False and spec['validation_launch_enabled'] is False
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
out=ROOT/'tracker_validation/contact_replay_v4_cpu_author_v1';assert not out.exists()
test=subprocess.run([sys.executable,str(ROOT/'tests/test_contact_replay_modes.py'),'--output',str(out)],env=env,capture_output=True,text=True,timeout=60)
assert test.returncode==0,(test.stdout,test.stderr)
disabled=[]
for mode in ['formal','validation']:
 run='disabled_contact_v4_cpu_'+mode
 result=subprocess.run([sys.executable,str(ROOT/'scripts/supervise_contact_filter_v4.py'),'--mode',mode,'--run',run,'--checkpoint','unused','--training-receipt','unused','--training-exit-receipt','unused'],env=env,capture_output=True,text=True,timeout=15)
 expected='Future exporter disabled; no SDK dispatched' if mode=='formal' else 'Future validation disabled; no SDK dispatched'
 assert result.returncode!=0 and expected in result.stderr
 assert not (ROOT/'tracker_validation'/run).exists()
 disabled.append({'mode':mode,'exit_code':result.returncode,'stderr':result.stderr,'sdk_dispatched':False})
sys.path.insert(0,str(ROOT/'scripts'))
from contact_replay_modes import authorize,select_validation_sources
training_path=ROOT/'tracker_validation/filter_policy_paper69_solid_floor_20261026_v1/receipt.json';raw=training_path.read_bytes();training=json.loads(raw)
try:authorize('validation',dict(spec,validation_launch_enabled=True),training,{'child_exit_code':0,'timed_out':False,'numerical_status':training['status']})
except AssertionError as error:rejection=str(error)
else:raise AssertionError('Active incomplete policy accepted')
rows=[];plan=json.loads((ROOT/'configs/filter-training-plan.json').read_text())
for name in plan['input_reference_exports']:
 folder=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references')/name
 receipt=json.loads((folder/'receipt.json').read_text());manifest=folder/'manifest.private.jsonl'
 assert sha(manifest)==receipt['private_manifest_sha256']
 rows.extend(json.loads(line) for line in manifest.read_text().splitlines() if not json.loads(line)['mirrored'])
fingerprint=hashlib.sha256(json.dumps([{'sha256':r['sha256'],'frames':r['frames']} for r in rows],sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert len(rows)==8278 and fingerprint==training['reference_fingerprint']==plan['contact_profile']['reference_order_fingerprint']
selected=select_validation_sources(rows,spec)
private=out/'validation_selection_plan.private.json'
private.write_text(json.dumps({'reference_fingerprint':fingerprint,'selected_original_sources':[{'index':i,'frames':rows[i]['frames'],'sha256':rows[i]['sha256']} for i in selected],'physics_executed':False},indent=2))
lock=(ROOT/'receipts/full_library_capacity.lock').open('r')
try:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);busy=False;fcntl.flock(lock,fcntl.LOCK_UN)
except BlockingIOError:busy=True
lock.close()
assert busy,'Active trainer should still own the common SDK lock'
assert all(sha(ROOT/n)==s for n,s in bound.items()) and all(sha(ROOT/n)==s for n,s in older.items())
result={'status':'CONTACT_REPLAY_V4_CPU_STAGING_VERIFIED_SDK_PENDING','test_receipt':json.loads((out/'receipt.json').read_text()),'test_exit_code':test.returncode,'test_stdout':test.stdout,'test_stderr':test.stderr,'disabled_supervisor_checks':disabled,'actual_training_rejected_for_validation':True,'actual_training_rejection':rejection,'actual_training_completed_iterations':training['completed_learning_iterations'],'actual_training_receipt_sha256':hashlib.sha256(raw).hexdigest(),'selection_metadata_verified':True,'reference_sources':len(rows),'reference_fingerprint':fingerprint,'selected_complete_sources':len(selected),'selected_frame_min':min(rows[i]['frames'] for i in selected),'selected_frame_max':max(rows[i]['frames'] for i in selected),'private_selection_plan_sha256':sha(private),'sdk_lock_observed_busy':busy,'frozen_runtime_files_verified':len(bound),'older_inactive_files_unchanged':len(older),'deployed':{n:v['sha256'] for n,v in payload.items()},'sdk_runs':0,'gpu_jobs':0,'physics_steps':0,'optimizer_updates':0,'physical_filter_executed':False,'dataset_accepted':False,'actual_sdk_validation_completed':False,'sample_unix':time.time()}
(out/'execution.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''.replace('PAYLOAD', repr(payload)).replace('BOUND', repr(bound)).replace('OLDER', repr(older))
transport = subprocess.run(SSH + ['python3 -'], input=script, capture_output=True,
                           text=True, encoding='utf-8', timeout=120)
result = json.loads(transport.stdout) if transport.stdout.strip() else {}
result.update(transport_exit_code=transport.returncode, transport_stderr=transport.stderr)
p = R / 'server_cpu_author_v1.json'
assert not p.exists()
p.write_text(json.dumps(result, indent=2), encoding='utf-8')
(R / 'candidate_runtime_bytes_v4.json').write_text(json.dumps({'files': payload}, indent=2), encoding='utf-8')
shutil.copy2(local, R / 'local_cpu_author_v1.json')
assert transport.returncode == 0, (transport.stderr, transport.stdout[-3000:])
private_dir = pathlib.Path('D:/具身智能/TextOp复现_20261024/datasets/contact_replay_v4_private_20261026')
private_dir.mkdir(parents=True, exist_ok=True)
private = private_dir / 'validation_selection_plan.private.json'
if not private.exists():
    transfer = subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-i',
        'C:/Users/Pende/.ssh/id_ed25519',
        'ouyanghao@10.147.17.32:/data/code/ouyanghao/oyh/textop-reproduction-20261024/tracker_validation/contact_replay_v4_cpu_author_v1/validation_selection_plan.private.json', str(private)],
        capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert transfer.returncode == 0, transfer.stderr
assert hashlib.sha256(private.read_bytes()).hexdigest() == result['private_selection_plan_sha256']
(R / 'private_selection_backup.json').write_text(json.dumps({'local_path': str(private),
    'sha256': result['private_selection_plan_sha256'], 'server_local_sha256_match': True,
    'mapping_publicly_uploaded': False}, indent=2), encoding='utf-8')
print(json.dumps({k: result[k] for k in ['status', 'actual_training_completed_iterations',
    'selected_complete_sources', 'selected_frame_min', 'selected_frame_max', 'frozen_runtime_files_verified',
    'older_inactive_files_unchanged', 'sdk_runs']}))

"""Validate actual contact witnesses and a fail-closed adapter before activation."""
import argparse,base64,hashlib,json,pathlib,sys
from session import remote
B=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--activate',action='store_true');a=p.parse_args()
assert a.activate,'Explicit already-authorized activation invocation required'
bound=['scripts/solid_floor_ppo_probe.py','scripts/supervise_solid_floor_ppo.py','scripts/solid_floor_diagnostic.py','scripts/paper_filter_config.py','scripts/filter_training_runtime.py','scripts/numerical_trace.py','scripts/audit_full_contact_state.py','configs/full-library-capacity-spec.json','configs/solid-floor-ppo-spec.json','configs/tracker-env-spec.json','configs/paper-step2.json','scripts/validated_contact_profile.py','scripts/train_filter_policy.py','scripts/supervise_filter_policy.py','scripts/checkpoint_storage_guard.py']
local={n:hashlib.sha256((B/n).read_bytes()).hexdigest() for n in bound}
script=r"""
import pathlib,json,hashlib,os,sys,types,copy,shutil,subprocess,time
root=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
local=LOCAL
for n,wanted in local.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==wanted,n
planpath=root/'configs/filter-training-plan.json';plan=json.loads(planpath.read_text());assert plan['formal_launch_enabled'] is False
oldsha=hashlib.sha256(planpath.read_bytes()).hexdigest()
profile={'kind':'validated_static_solid_floor','enabled':True,'training_math_changes':False,'self_collisions':True,'bound_files':local,'witnesses':[],
'reference_order_fingerprint':'0b786ba08bf8f33aa92cb4e91ed72cd5b56159c4e251cedf88f429269141c1f7',
'backup_audit_sha256':'8c742cdbeef4773bfc66fea828fb68d9c91d70f656e9d9385a70c31241fe307e'}
for run in ['solid_floor_ppo_author_v1','solid_floor_ppo_document_v2']:
 out=root/'tracker_validation'/run;r=out/'receipt.json';e=root/'receipts'/('full_contact_'+run+'_exit.json')
 data=json.loads(r.read_text());exit=json.loads(e.read_text())
 assert exit['child_exit_code']==0 and not exit['timed_out']
 profile['witnesses'].append({'run':run,'receipt_sha256':hashlib.sha256(r.read_bytes()).hexdigest(),'exit_sha256':hashlib.sha256(e.read_bytes()).hexdigest(),'trace_sha256':data['numerical_trace']['trace_sha256']})
# Read-only evidence/guard checks. Stub only the already-GPU-validated floor factory,
# so this CPU test does not claim another physics or geometry validation.
sys.path.insert(0,str(root/'scripts'));from validated_contact_profile import apply
calls=[]
def fake_create(path):calls.append(str(path));return str(path)
sys.modules['solid_floor_diagnostic']=types.SimpleNamespace(create=fake_create)
def cfg(self_collision=True):return types.SimpleNamespace(scene=types.SimpleNamespace(robot=types.SimpleNamespace(spawn=types.SimpleNamespace(articulation_props=types.SimpleNamespace(enabled_self_collisions=self_collision))),terrain=types.SimpleNamespace(usd_path=None)),sim=types.SimpleNamespace(dt=0.005),decimation=4)
results=[]
c=cfg();result=apply(c,root/'tracker_validation/formal_contact_binding_cpu_v1',root,profile);assert len(calls)==1
results.append('Actual two-witness receipts and sources accepted; factory dispatch reached once (stubbed)')
for kind in ['source_hash','receipt_hash','trace_hash','reference_order','duplicate_witness','self_collision']:
 trial=copy.deepcopy(profile);config=cfg()
 if kind=='source_hash':trial['bound_files']['scripts/solid_floor_diagnostic.py']='0'*64
 elif kind=='receipt_hash':trial['witnesses'][0]['receipt_sha256']='0'*64
 elif kind=='trace_hash':trial['witnesses'][0]['trace_sha256']='0'*64
 elif kind=='reference_order':trial['reference_order_fingerprint']='0'*64
 elif kind=='duplicate_witness':trial['witnesses'][1]=copy.deepcopy(trial['witnesses'][0])
 elif kind=='self_collision':config=cfg(False)
 try:apply(config,root/'tracker_validation/formal_contact_binding_cpu_v1',root,trial)
 except AssertionError:results.append(kind+' rejected')
 else:raise AssertionError('Tampered binding accepted: '+kind)
assert len(calls)==1 and len(results)==7
from checkpoint_storage_guard import check,schedule
storage=check(root,root,plan['checkpoint_storage'],len(schedule(plan['learning_iterations'],plan['checkpoint_interval'])))
cap=json.loads((root/'configs/full-library-capacity-spec.json').read_text())
mem=[int(x.split()[1])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0];assert mem>=cap['minimum_available_host_ram_gib']*1024**3
free={int(x.split(',')[0]):int(x.split(',')[1])*1024**2 for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()};assert free[1]>=plan['minimum_gpu_free_gib']*1024**3
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 assert not {str(root/'scripts'/x) for x in ['run_tracker.sh','solid_floor_ppo_probe.py','train_filter_policy.py']}.intersection(argv),'Existing SDK job'
result={'status':'CONTACT_BINDING_CPU_GUARDS_AND_STORAGE_VERIFIED','checks':results,'gpu_calls':0,'physics_calls':0,'floor_factory_stubbed':True,'old_plan_sha256':oldsha,'profile':profile,'storage':storage,'host_available_bytes':mem,'gpu_free_bytes':free,'time':time.time()}
proof=root/'receipts/formal_contact_binding_cpu_v1.json';assert not proof.exists();proof.write_text(json.dumps(result,indent=2));print(json.dumps(result))
""".replace('LOCAL',repr(local))
r=json.loads(remote(script,timeout=180));R=B/'results/full_contact_20261026/formal_launch';R.mkdir(exist_ok=True)
(R/'binding_cpu_storage_author.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
planpath=B/'configs/filter-training-plan.json';old=planpath.read_bytes();assert hashlib.sha256(old).hexdigest()==r['old_plan_sha256']
(R/'plan_before_activation.json').write_bytes(old)
plan=json.loads(old);plan.update(formal_launch_enabled=True,formal_launch_blockers=[],contact_profile=r['profile'])
plan['completed_prerequisites'].append('Static solid-floor author and independent document_v2 each10actualPPOiterations/200updates, no diagnostic alerts/nonfinite trace; exact underlying simulator cause and long-run convergence not claimed')
plan['latest_review']={'review':'contact representation validated with two actual runs and byte-bound CPU guard/storage tests','formal_training_started':False,'long_run_convergence_established':False}
raw=(json.dumps(plan,indent=2)+'\n').encode('utf-8')
# Preserve exact bytes and compare old remote plan before activating.
script=r"""import pathlib,hashlib,base64,json
p=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024/configs/filter-training-plan.json')
assert hashlib.sha256(p.read_bytes()).hexdigest()==OLD
backup=p.parent.parent/'receipts/filter_training_plan_before_solid_floor_activation.json';assert not backup.exists();backup.write_bytes(p.read_bytes())
raw=base64.b64decode(DATA);p.write_bytes(raw);print(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'formal_launch_enabled':json.loads(p.read_text())['formal_launch_enabled']}))
""".replace('OLD',repr(r['old_plan_sha256'])).replace('DATA',repr(base64.b64encode(raw).decode()))
deployed=json.loads(remote(script,timeout=60));assert deployed['sha256']==hashlib.sha256(raw).hexdigest();planpath.write_bytes(raw)
activation={'launch_enabled':True,'run':'filter_policy_paper69_solid_floor_20261026_v1','deployed_files':dict(local,**{'configs/filter-training-plan.json':deployed['sha256']}),'actual_witnesses':[x['run'] for x in r['profile']['witnesses']],'checkpoint_startup_budget':r['storage'],'formal_training_started':False,'physical_filter_executed':False}
(B/'configs/formal-contact-activation.json').write_text(json.dumps(activation,indent=2),encoding='utf-8')
(R/'activation.json').write_text(json.dumps(activation,indent=2),encoding='utf-8');print(json.dumps({'activated':True,'plan_sha256':deployed['sha256'],'checks':len(r['checks']),'training_started':False}))

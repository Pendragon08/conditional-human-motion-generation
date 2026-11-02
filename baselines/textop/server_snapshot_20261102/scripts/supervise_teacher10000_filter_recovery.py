"""Continue the10k teacher after its all-source relaxed5percent gate is empty.

Original PPO and40k reference budget stay frozen. Inspect each new5k checkpoint;
no generator may train until actual filtering and corpus acceptance.
"""
import argparse,fcntl,hashlib,json,os,pathlib,subprocess,time
from checkpoint_storage_guard import check
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources

ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
sp=ROOT/'configs/teacher10000-filter-recovery-spec.json';s=json.loads(sp.read_text())
assert os.getuid()==1013 and (a.preflight_only or s['resume_enabled'] is True)
assert s['schema']=='textop.teacher10000_filter_recovery.v1'
assert sha(__file__)==s['source_sha256']
lock=(ROOT/'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for name,h in s['frozen_bindings'].items():assert sha(ROOT/name)==h,name
for item in s['evidence']:
    assert sha(ROOT/item['path'])==item['sha256'],item['path']
accept=json.loads((ROOT/s['functional_acceptance']).read_text())
assert accept['functional_chain_accepted'] and accept['quality_review_completed']
assert not accept['paper_quality_accepted'] and not accept['paper_complete']
assert accept['full_training_research_continuation_authorized']
for name in accept['online_runs']:
    o=ROOT/'online_private'/name
    j=json.loads((o/'sdk/receipt.json').read_text())
    ex=json.loads((ROOT/'receipts'/('online_'+name+'_outer_exit.json')).read_text())
    assert ex['actual_returncode']==0 and j['writers_closed'] and j['output_writers_closed']
    assert j['actual_control_calls']==257 and j['actual_physics_substeps']==1028
    assert j['actual_50hz_deadlines_passed'] and j['deadline_misses']==0
    assert not j['physical_failure'] and j['stream_underruns']==0 and j['switched_target_consumed']
cp=pathlib.Path(s['checkpoint']['path'])
assert cp.stat().st_uid==1013 and not cp.is_symlink() and sha(cp)==s['checkpoint']['sha256']
assert s['checkpoint']['completed']==10000 and s['checkpoint']['updates']==200000
assert s['target_completed_iterations']==40000
from teacher10000_pilot_protocol import load_teacher,authorize_teacher
prior_spec=json.loads((ROOT/'configs/teacher10000-pilot-spec.json').read_text())
prior=load_teacher(ROOT,prior_spec);meta=authorize_teacher(prior_spec,prior)
assert meta['sha256']==s['checkpoint']['sha256'] and pathlib.Path(meta['path'])==cp
paused=ROOT/'receipts/generator8278_user_pause_verified_20261027_v1.json'
assert sha(paused)==s['generator_pause_acceptance_sha256']
pj=json.loads(paused.read_text());assert pj['status']=='PREFILTER_GENERATOR_PAUSED_CHECKPOINT_PRIVATE_BACKUP_VERIFIED' and pj['real_worker_exited']
relaxed=ROOT/'teacher10000_full_filter_20261027/relaxed_scope_v1/summary.json'
assert sha(relaxed)==s['relaxed_scope_sha256']
rj=json.loads(relaxed.read_text());assert rj['source_count']==8278 and rj['counts']['eligible_for_strict']==0 and not rj['strict_evaluation_completed']
plan=json.loads((ROOT/'configs/filter-training-plan.json').read_text())
assert plan['learning_iterations']==40000 and plan['num_envs']==8192 and plan['formal_launch_enabled']
for proc in pathlib.Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
    try:
        if proc.stat().st_uid!=1013:continue
        argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
    except (OSError,ProcessLookupError):continue
    blocked={'run_tracker.sh','train_filter_policy.py','online_quick_sdk.py','online_quick_generator.py',
             'run_quick_vae.py','run_quick_ldm.py','run_generated_pipeline_sdk.py','run_generator8278_vae.py'}
    assert not any(x==str(ROOT/'scripts'/name) for name in blocked for x in argv),'Active personal experiment'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
free={int(x.split(',')[0]):int(x.split(',')[1])*1024**2 for x in gpu.strip().splitlines()}
assert free[1]>=20*1024**3
resources=sample_resources(ROOT,None,20*1024**3,2*1024**3)
budget=check(ROOT,ROOT,plan['checkpoint_storage'],9)
run=s['run'];out=ROOT/'tracker_validation'/run
assert not out.exists(),'Never overwrite or restart an existing run'
prefix='filter_recovery_'+run
cmd=['bash',str(ROOT/'scripts/run_tracker.sh'),str(ROOT/'scripts/train_filter_policy.py'),
     '--spec',str(ROOT/'configs/tracker-env-spec.json'),'--plan',str(ROOT/'configs/filter-training-plan.json'),
     '--output',str(out),'--mode','formal','--resume',str(cp),'--device','cuda:1','--headless']
pre={'status':'FULL_TEACHER_CONTINUATION_PREFLIGHT_PASSED','run':run,'spec_sha256':sha(sp),
     'checkpoint_sha256':sha(cp),'resumed_completed':10000,'resumed_updates':200000,
     'target_completed':40000,'target_updates':800000,'startup_budget':budget,'resources':resources,
     'gpu_free_bytes':free,'command':cmd,'full_generator_training_started':False,
     'physics_restart_not_bit_exact_continuation':True,'paper_quality_accepted':False}
print(json.dumps(pre),flush=True)
if a.preflight_only:raise SystemExit(0)
launch=ROOT/'receipts'/(prefix+'_launch.json');assert not launch.exists()
with launch.open('x') as f:json.dump(dict(pre,supervisor_pid=os.getpid(),started_unix=time.time()),f,indent=2)
def sample(pid):
    value=sample_resources(ROOT,pid,2*1024**3,2*1024**3)
    live=ROOT/'receipts'/(prefix+'_live.json');tmp=live.with_suffix('.partial')
    tmp.write_text(json.dumps({'child_pid':pid,'sampled_unix':time.time(),'resources':value}));tmp.replace(live)
    return value
proof=run_supervised(cmd,ROOT/'logs'/(prefix+'.log'),ROOT/'logs'/(prefix+'_resources.jsonl'),
                     ROOT/'receipts'/(prefix+'_process_exit.json'),sample,7*86400,60)
receipt=json.loads((out/'receipt.json').read_text()) if (out/'receipt.json').exists() else {}
proof.update(run=run,spec_sha256=sha(sp),receipt_sha256=sha(out/'receipt.json') if receipt else None,
             completed=receipt.get('completed_learning_iterations'),updates=receipt.get('optimizer_updates'),
             full_generator_training_started=False)
with (ROOT/'receipts'/(prefix+'_exit.json')).open('x') as f:json.dump(proof,f,indent=2)
assert proof['supervision_passed'] and receipt['status']=='TEXTOP_FORMAL_FILTER_TRAINING_FINISHED'
assert proof['completed']==40000 and proof['updates']==800000 and receipt['output_writers_closed']

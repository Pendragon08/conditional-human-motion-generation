import pathlib,json,hashlib,os,sys,fcntl,subprocess,shutil,time
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
assert os.getuid()==1013 and ROOT.resolve()==ROOT
os.sched_setaffinity(0,{0,1,2,3})
sys.path.insert(0,str(ROOT/'scripts'))
from quick_pilot_protocol import load_teacher,authorize_teacher,settings_signature,STATUS
from strict_tracking_coverage import verify_trial
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
specpath=ROOT/'configs/quick-pilot-spec.json';spec=json.loads(specpath.read_text())
assert spec['pilot_launch_enabled'] is True and spec['sample_launch_enabled'] is False
assert spec['device']=='cuda:1' and spec['torch_memory_fraction']==0.65 and spec['cpu_threads']==4
assert spec['smoke_parallel_envs']==8 and spec['smoke_max_control_calls']==1024 and spec['smoke_timeout_seconds']==600
teacher=load_teacher(ROOT,spec);ck=authorize_teacher(spec,teacher)
assert sha(ck['path'])==ck['sha256']
source_hashes={}
for rel,value in teacher['quick_plan']['frozen_runtime_files'].items():
    assert sha(ROOT/rel)==value
    source_hashes[rel]=value
for name,key in [('evaluate_quick_pilot.py','executor_sha256'),('quick_pilot_protocol.py','protocol_sha256'),('quick_pilot_trace.py','trace_sha256'),('supervise_quick_pilot.py','supervisor_sha256'),('contact_process_supervision.py','supervision_sha256'),('contact_filter_resources.py','resource_sampler_sha256'),('strict_tracking_coverage.py','coverage_sha256')]:
    assert sha(ROOT/'scripts'/name)==spec[key]
    source_hashes['scripts/'+name]=spec[key]
run='quick_pilot_smoke_author_v1';out=ROOT/'tracker_validation'/run;prefix='quick_pilot_'+run
paths={'receipt':out/'receipt.json','exit':ROOT/'receipts'/(prefix+'_exit.json'),'process_exit':ROOT/'receipts'/(prefix+'_process_exit.json'),'resources':ROOT/'logs'/(prefix+'_resources.jsonl')}
data={key:json.loads(paths[key].read_text()) for key in ['receipt','exit','process_exit']}
r,e,pr=[data[key] for key in ['receipt','exit','process_exit']]
assert r['status']==STATUS and r['stage']=='smoke' and r['mode']=='quick_pilot'
assert r['output_writers_closed'] is True and r['optimizer_updates']==0
assert r['dataset_accepted'] is False and r['physical_filter_executed'] is False
assert r['policy_checkpoint_sha256']==ck['sha256'] and r['policy_checkpoint_completed_iterations']==5000
assert r['source_sha256']==spec['executor_sha256'] and r['execution_settings_sha256']==settings_signature(spec)
assert r['quick_protocol_sha256']==spec['protocol_sha256'] and r['trace_source_sha256']==spec['trace_sha256']
assert r['pilot_spec_sha256']==sha(specpath)
assert r['terminal_execution_state_captured_before_autoreset'] is True
assert r['replayed_original_sources']==r['finished_trials']==8 and r['complete_trials']>=1
assert r['complete_trials']+r['failed_trials']==8
assert 0<r['actual_vectorized_control_calls']<=1024 and r['actual_scene_physics_steps']==4*r['actual_vectorized_control_calls']
assert e['run']==run and e['supervision_passed'] is True and e['child_exit_code']==0 and e['timed_out'] is False
assert e['evaluation_receipt_sha256']==sha(paths['receipt']) and e['numerical_status']==STATUS
assert e['resources_sha256']==sha(paths['resources']) and e['process_exit_sha256']==sha(paths['process_exit'])
for key in ['child_pid','child_exit_code','child_reaped','timed_out','supervision_passed','supervisor_error','samples_written']:
    assert pr[key]==e[key]
assert pr['child_reaped'] is True and pr['supervisor_error'] is None and not pr.get('cleanup_error') and pr['sample_errors']==[]
sr=[json.loads(s) for s in paths['resources'].read_text().splitlines()]
assert len(sr)==pr['samples_written'] and len(sr)>=3
assert sr[0]['phase']=='before_launch' and sr[1]['phase']=='after_launch' and sr[-1]['phase']=='after_exit'
assert sr[-1]['owned_group_processes']==[]
assert all(all(p['cpu_affinity']==[0,1,2,3] for p in s['owned_group_processes']) for s in sr)
strict=out/'strict.private.json';journal=out/'strict_trials.private.jsonl';index=out/'executed_traces.private.json'
assert sha(strict)==r['private_strict_log_sha256'] and sha(journal)==r['private_trial_journal_sha256']
assert sha(index)==r['private_executed_trace_index_sha256']
log=json.loads(strict.read_text());assert log['diagnostic_only'] is True and len(log['motions'])==8
expected=[];natural=failed=0
for row in log['motions']:
    assert len(row['trials'])==1
    trial=row['trials'][0];verify_trial(trial,trial['reference_frames'],0)
    expected.append({'motion_id':row['motion_id'],'trial':trial})
    natural+=int(trial['complete']);failed+=int(trial['failed'])
encode=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'))
actual=[json.loads(line) for line in journal.read_text().splitlines()]
assert sorted(map(encode,expected))==sorted(map(encode,actual))
records=json.loads(index.read_text());assert len(records)==8
ids=[];trace_samples=0
for record in records:
    folder=pathlib.Path(record['folder']);assert folder.resolve().parent==out/'execution_private'
    meta=json.loads((folder/'trace.private.json').read_text())
    assert meta['status']=='FINISHED_PHYSICAL_TRIAL' and meta['runtime_kind']=='isaaclab_physics'
    assert meta['phase']=='INITIAL_THEN_PRE_AUTORESET_TERMINATION_COMPUTE'
    assert meta['samples']==meta['trial']['control_steps']+1==record['samples']
    assert sha(folder/'poses.private.bin')==meta['binary_sha256']
    assert sha(folder/'executed_poses.private.npz')==meta['npz_sha256']==record['npz_sha256']
    ids.append(meta['motion_id']);trace_samples+=meta['samples']
assert sorted(ids)==sorted(row['motion_id'] for row in log['motions'])
assert natural==r['complete_trials'] and failed==r['failed_trials']
assert not (out/'evidence_inputs.private.json').exists()
blocked=['run_tracker.sh','train_filter_policy.py','evaluate_strict_filter.py','evaluate_strict_filter_contact.py','evaluate_contact_filter_v4.py','evaluate_quick_pilot.py']
owned=[];conflicts=[];stopped=[]
for proc in pathlib.Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
    try:
        if proc.stat().st_uid!=1013:continue
        argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        state=next(x for x in (proc/'status').read_text().splitlines() if x.startswith('State:'))
        entry={'pid':int(proc.name),'argv':[x for x in argv if x],'state':state}
        owned.append(entry)
        if {str(ROOT/'scripts'/x) for x in blocked}.intersection(argv):conflicts.append(entry)
        if 'T' in state.split()[1]:stopped.append({'pid':int(proc.name),'state':state,'argv':[x for x in argv if x]})
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
assert not conflicts,conflicts
oldpid=teacher['stop']['trainer_pid'];old=pathlib.Path('/proc')/str(oldpid)
old_state={'pid':oldpid,'exists':old.exists(),'original_start_ticks':teacher['stop']['trainer_start_ticks']}
if old.exists():
    raw=(old/'stat').read_text();tick=int(raw[raw.rfind(')')+2:].split()[19]);old_state['current_start_ticks']=tick
    assert tick!=teacher['stop']['trainer_start_ticks'],'Original trainer still present'
lockpath=ROOT/'receipts/full_library_capacity.lock'
with lockpath.open('r+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock_available=True
    fcntl.flock(lock,fcntl.LOCK_UN)
mem=[int(x.split()[1])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0]
cap=json.loads((ROOT/'configs/full-library-capacity-spec.json').read_text())
assert mem>=cap['minimum_available_host_ram_gib']*1024**3
disk=shutil.disk_usage(ROOT);assert disk.free>spec['minimum_preflight_disk_free_gib']*1024**3
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
free={int(x.split(',')[0]):int(x.split(',')[2]) for x in gpu.strip().splitlines()};assert free[1]>20*1024
assert int(next(x for x in gpu.splitlines() if int(x.split(',')[0])==1).split(',')[1])<500
newrun='quick_pilot_smoke_document_v1';newprefix='quick_pilot_'+newrun
assert not (ROOT/'tracker_validation'/newrun).exists()
for p in [ROOT/'receipts'/(newprefix+'_exit.json'),ROOT/'receipts'/(newprefix+'_process_exit.json'),ROOT/'logs'/(newprefix+'.log'),ROOT/'logs'/(newprefix+'_resources.jsonl')]:assert not p.exists()
report={'status':'PREFLIGHT_PASSED','unix_time':time.time(),'root':str(ROOT),'uid':os.getuid(),'author':data,'author_hashes':{k:sha(v) for k,v in paths.items()},'author_private_verified':{'closed_trials':8,'natural_complete':natural,'failed':failed,'trace_samples':trace_samples,'binary_npz_and_journal_hashes_verified':True,'raw_private_data_downloaded':False},'teacher_artifact_sha256':teacher['artifact_sha256'],'teacher_checkpoint':ck,'teacher_stop':teacher['stop'],'original_trainer_current_state':old_state,'source_sha256':source_hashes,'spec_sha256':sha(specpath),'warm_environment_canonical_hash':spec['warm_environment_canonical_hash'],'resources':{'available_ram_bytes':mem,'persistent_free_bytes':disk.free,'nvidia_smi':gpu.strip(),'free_gpu1_mib':free[1]},'common_lock_available':lock_available,'sdk_conflicts':conflicts,'preserved_stopped_processes':stopped,'owned_processes':owned,'fresh_document_outputs_absent':True}
print(json.dumps(report,ensure_ascii=False))

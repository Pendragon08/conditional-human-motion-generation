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
run='quick_pilot_smoke_document_v1';out=ROOT/'tracker_validation'/run;prefix='quick_pilot_'+run
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
# Existing explicit user authorization permits shared GPU1; supervisor free>20GiB and Torch0.65 remain binding.

# Read private binary and NPZ payloads in place using stdlib only. No SDK or package imports.
import ast,struct,zipfile,math
def npy(payload):
    assert payload[:6]==b'\x93NUMPY'
    major=payload[6]
    if major==1:
        length=struct.unpack('<H',payload[8:10])[0];offset=10
    else:
        assert major in [2,3]
        length=struct.unpack('<I',payload[8:12])[0];offset=12
    header=ast.literal_eval(payload[offset:offset+length].decode('utf-8').strip())
    assert header['fortran_order'] is False
    return header,payload[offset+length:]
payload_count=0;binary_bytes=npz_bytes=all_samples=0;sample_lengths=[];per_trial_steps=[]
for record in records:
    folder=pathlib.Path(record['folder']);meta=json.loads((folder/'trace.private.json').read_text())
    binary=folder/'poses.private.bin';compressed=folder/'executed_poses.private.npz'
    raw=binary.read_bytes();count=meta['samples'];width=meta['row_width']
    assert len(raw)==count*width*4
    floats=struct.unpack('<'+'f'*(count*width),raw)
    assert all(math.isfinite(v) for v in floats)
    with zipfile.ZipFile(compressed) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={f['name']+'.npy' for f in meta['fields']}|{'time_seconds.npy','joint_names.npy','runtime_kind.npy','capture_phase.npy'}
        offset=0
        for field in meta['fields']:
            h,payload=npy(z.read(field['name']+'.npy'))
            shape=tuple(field['shape']);size=math.prod(shape)
            assert h['descr']=='<f4' and h['shape']==(count,)+shape
            values=struct.unpack('<'+'f'*(count*size),payload)
            expected=tuple(floats[i*width+offset+j] for i in range(count) for j in range(size))
            assert values==expected
            if field['name']=='reference_frame':
                assert values==(0.0,)+tuple(float(i) for i in range(count-1))
            if field['name']=='root_quat_wxyz':
                assert all(.99<math.sqrt(sum(values[i*4+j]**2 for j in range(4)))<1.01 for i in range(count))
            offset+=size;payload_count+=1
        assert offset==width
        h,payload=npy(z.read('time_seconds.npy'))
        assert h['descr']=='<f8' and h['shape']==(count,)
        times=struct.unpack('<'+'d'*count,payload)
        assert times==tuple(i/50. for i in range(count))
        for name,expected in [('runtime_kind',meta['runtime_kind']),('capture_phase',meta['phase'])]:
            h,payload=npy(z.read(name+'.npy'));assert h['shape']==() and h['descr'].startswith('<U')
            assert payload.decode('utf-32-le').rstrip('\0')==expected
        h,payload=npy(z.read('joint_names.npy'))
        assert h['shape']==(29,) and h['descr'].startswith('<U')
        size=int(h['descr'][2:])*4
        names=[payload[i*size:(i+1)*size].decode('utf-32-le').rstrip('\0') for i in range(29)]
        assert names==meta['joint_names']
    assert meta['final_reference_endpoint_comparison_performed'] is False
    assert meta['consumed_reference_frame_max']==meta['trial']['control_steps']-1
    assert meta['reference_final_frame']==meta['reference_frames']-1
    assert meta['samples']==meta['reference_frames'] if meta['trial']['complete'] else meta['samples']<=meta['reference_frames']
    all_samples+=count;binary_bytes+=len(raw);npz_bytes+=compressed.stat().st_size
    sample_lengths.append(count);per_trial_steps.append(meta['trial']['control_steps'])
assert len(records)==r['finished_trials']==8 and r['completion_branch_exercised'] is True
assert max(per_trial_steps)==r['actual_vectorized_control_calls']
assert binary_bytes+npz_bytes==r['private_trace_bytes']
assert not list(out.rglob('*.partial'))
open_private_fds=[]
for proc in pathlib.Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        if proc.stat().st_uid!=os.getuid():continue
        for fd in (proc/'fd').iterdir():
            try:
                target=os.readlink(fd)
                if target.startswith(str(out)+'/'):open_private_fds.append({'pid':int(proc.name),'fd':fd.name})
            except OSError:pass
    except (OSError,ProcessLookupError):pass
assert not open_private_fds
private_payload_audit={'npz_binary_arrays_exact':True,'finite_pose_action_and_target_values':True,'pre_autoreset_capture_phase_verified':True,'initial_then_consumed_reference_sequence_verified':True,'time_seconds_exact_50hz':True,'joint_names_exact':True,'npz_crc_passed':True,'payload_array_count':payload_count,'closed_trace_count':len(records),'total_samples':all_samples,'sample_lengths_sorted':sorted(sample_lengths),'per_trial_control_steps_sorted':sorted(per_trial_steps),'private_binary_bytes':binary_bytes,'private_npz_bytes':npz_bytes,'partial_files':0,'open_owned_private_fds':open_private_fds,'private_arrays_downloaded':False,'final_reference_endpoint_comparison_performed':False}

report={'status':'DOCUMENT_ACTUAL_AUDIT_PASSED','unix_time':time.time(),'root':str(ROOT),'uid':os.getuid(),'document':data,'document_hashes':{k:sha(v) for k,v in paths.items()},'document_private_verified':{'closed_trials':8,'natural_complete':natural,'failed':failed,'trace_samples':trace_samples,'binary_npz_and_journal_hashes_verified':True,'raw_private_data_downloaded':False},'teacher_artifact_sha256':teacher['artifact_sha256'],'teacher_checkpoint':ck,'teacher_stop':teacher['stop'],'original_trainer_current_state':old_state,'source_sha256':source_hashes,'spec_sha256':sha(specpath),'warm_environment_canonical_hash':spec['warm_environment_canonical_hash'],'resources':{'available_ram_bytes':mem,'persistent_free_bytes':disk.free,'nvidia_smi':gpu.strip(),'free_gpu1_mib':free[1]},'common_lock_available':lock_available,'sdk_conflicts':conflicts,'preserved_stopped_processes':stopped,'owned_processes':owned,'document_outputs_exist':True,'private_payload_audit':private_payload_audit}
print(json.dumps(report,ensure_ascii=False))

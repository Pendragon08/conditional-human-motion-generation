import pathlib,sys,json,time,hashlib
W=pathlib.Path(__file__).resolve().parent
B=pathlib.Path('C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop')
sys.path.insert(0,str(B/'provider_tools'))
from session import remote
script=r'''
import pathlib,json,subprocess,os,time,hashlib,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
run='quick_pilot_smoke_author_v1';out=r/'tracker_validation'/run;prefix='quick_pilot_'+run
paths={'receipt':out/'receipt.json','exit':r/'receipts'/(prefix+'_exit.json'),'process_exit':r/'receipts'/(prefix+'_process_exit.json'),'resources':r/'logs'/(prefix+'_resources.jsonl'),'stop':r/'receipts/quick5000_controlled_stop.json','backup':r/'receipts/quick5000_backup_verified.json','spec':r/'configs/quick-pilot-spec.json'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
data={k:json.loads(v.read_text()) for k,v in paths.items() if k!='resources'}
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
appdata=[]
for line in apps.stdout.splitlines():
    f=[x.strip() for x in line.split(',')]
    if len(f)!=3 or not f[0].isdigit():continue
    p=pathlib.Path('/proc')/f[0];v={'pid':int(f[0]),'gpu_uuid':f[1],'memory_mib':f[2]}
    try:
        v['uid']=p.stat().st_uid
        v['command_name']=(p/'comm').read_text().strip()
        v['state']=next(x for x in (p/'status').read_text().splitlines() if x.startswith('State:'))
        v['affinity']=sorted(os.sched_getaffinity(int(f[0])))
    except (OSError,ProcessLookupError) as error:v['lookup_error']=repr(error)
    appdata.append(v)
owned=[]
for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        if p.stat().st_uid!=1013:continue
        argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
        if any(x.endswith(('.py','.sh')) for x in argv):owned.append({'pid':int(p.name),'argv':[x for x in argv if x],'state':next(x for x in (p/'status').read_text().splitlines() if x.startswith('State:'))})
    except (OSError,ProcessLookupError):pass
newrun='quick_pilot_smoke_document_v1';newprefix='quick_pilot_'+newrun
newpaths=[r/'tracker_validation'/newrun,r/'receipts'/(newprefix+'_exit.json'),r/'receipts'/(newprefix+'_process_exit.json'),r/'logs'/(newprefix+'.log'),r/'logs'/(newprefix+'_resources.jsonl')]
print(json.dumps({'captured_unix':time.time(),'author':data,'hashes':{k:sha(v) for k,v in paths.items()},'author_resources':[json.loads(s) for s in paths['resources'].read_text().splitlines()],'gpu':{'exit_code':gpu.returncode,'stdout':gpu.stdout,'stderr':gpu.stderr},'compute_apps':{'exit_code':apps.returncode,'stdout':apps.stdout,'stderr':apps.stderr,'process_metadata':appdata},'available_ram_line':next(x for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')),'persistent_free_bytes':shutil.disk_usage(r).free,'owned_script_processes':owned,'document_outputs':{str(p):p.exists() for p in newpaths}},ensure_ascii=False))
'''
raw=remote(script,timeout=60)
(W/'failure_state.json').write_text(raw,encoding='utf-8')
j=json.loads(raw)
report={'status':'DOCUMENT_SDK_NOT_LAUNCHED_PREFLIGHT_FAILED','failure':'run-experiment SKILL preflight free GPU criterion memory.used < 500 MiB failed','exact_assertion':'assert int(next(x for x in gpu.splitlines() if int(x.split(\",\")[0])==1).split(\",\")[1])<500','preflight_remote_line':105,'actual_document_command':'python3 scripts/supervise_quick_pilot.py --stage smoke --run quick_pilot_smoke_document_v1','actual_document_exit':None,'document_trials':0,'no_retry_performed':True,'no_shared_source_or_configuration_changes':True,'private_motion_arrays_downloaded':False,'author_summary':{k:j['author']['receipt'].get(k) for k in ['finished_trials','complete_trials','failed_trials','actual_vectorized_control_calls','actual_scene_physics_steps','output_writers_closed','native_close_returned','teardown_fallback_used']},'author_child_exit_code':j['author']['exit']['child_exit_code'],'author_supervision_passed':j['author']['exit']['supervision_passed'],'author_hashes':j['hashes'],'gpu_failure_capture':j['gpu'],'gpu_app_owners':j['compute_apps']['process_metadata'],'document_outputs_absent':all(v is False for v in j['document_outputs'].values())}
(W/'verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))

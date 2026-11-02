import pathlib,sys,json,subprocess,time,hashlib,traceback
W=pathlib.Path(__file__).resolve().parent
B=pathlib.Path('C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop')
sys.path.insert(0,str(B/'provider_tools'))
from session import SSH
root='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
documented='python3 scripts/supervise_quick_pilot.py --stage smoke --run quick_pilot_smoke_document_v1'
assert documented in (W/'QUICK_PILOT_ENVIRONMENT_at_start.md').read_text(encoding='utf-8')
pre=json.loads((W/'preflight_v2.json').read_text(encoding='utf-8'))
assert pre['status']=='PREFLIGHT_PASSED' and pre['common_lock_available'] is True and pre['fresh_document_outputs_absent'] is True
assert pre['local_and_remote_bound_sources_match'] is True
assert not (W/'document_launch.json').exists() and not (W/'document_stdout.log').exists()
started=time.time();record={'documented_command':documented,'remote_working_directory':root,'exact_invocation_verbatim':True,'started_unix':started,'transport':'provider_tools/session.py SSH','preflight_sha256':hashlib.sha256((W/'preflight_v2.json').read_bytes()).hexdigest()}
with (W/'document_stdout.log').open('xb') as stdout,(W/'document_stderr.log').open('xb') as stderr:
    child=subprocess.Popen(SSH+['cd '+root+' && '+documented],stdout=stdout,stderr=stderr)
    record['local_ssh_pid']=child.pid
    with (W/'document_launch.json').open('x',encoding='utf-8') as target:json.dump(record,target,indent=2)
    try:
        code=child.wait(timeout=750)
        record.update(actual_ssh_exit=code,local_ssh_reaped=True,finished_unix=time.time(),elapsed_seconds=time.time()-started)
    except BaseException:
        record.update(observer_error=traceback.format_exc(),finished_unix=time.time(),blind_retry_performed=False)
        with (W/'document_execution.json').open('x',encoding='utf-8') as target:json.dump(record,target,indent=2)
        raise
for name in ['document_stdout.log','document_stderr.log']:
    record[name+'_sha256']=hashlib.sha256((W/name).read_bytes()).hexdigest()
with (W/'document_execution.json').open('x',encoding='utf-8') as target:json.dump(record,target,indent=2)
print(json.dumps(record))
assert code==0,'Documented command returned nonzero; preserve exact state without retry'

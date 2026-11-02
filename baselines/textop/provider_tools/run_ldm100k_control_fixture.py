"""Run the new CPU-only control fixture with retained real child/stream outcomes."""
import argparse,ast,hashlib,json,pathlib
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1];O=B/'results/ldm100k_user_handoff_20261028'
p=argparse.ArgumentParser();p.add_argument('--run',choices=['author_v1','document_v1'],required=True);a=p.parse_args()
code=r'''
import hashlib,json,os,pathlib,shutil,subprocess,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();sp=r/'configs/ldm100k-user-control-spec.json';s=json.loads(sp.read_bytes());assert not s['control_authorized']
names=list(s['bound_sources'])+['scripts/watch_ldm100k_handoff.py','scripts/supervise_ldm100k_handoff.py','provider_tools/test_ldm100k_control_cpu.py','configs/ldm100k-user-control-spec.json']
before={n:sha(r/n) for n in names};assert shutil.disk_usage(r).free>=2*1024**3
assert int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024>1024**3
root=r/'ldm100k_user_handoff_20261028';root.mkdir(mode=0o700,exist_ok=True);assert root.stat().st_uid==1013
run='control_fixture_'+RUN;out=root/run;stdout=root/(run+'.stdout.txt');stderr=root/(run+'.stderr.txt')
assert not any(x.exists() for x in [out,stdout,stderr])
with stdout.open('xb') as o,stderr.open('xb') as e:
 q=subprocess.Popen(['python3',str(r/'provider_tools/test_ldm100k_control_cpu.py'),'--output',str(out)],stdout=o,stderr=e,start_new_session=True)
 try:code=q.wait(timeout=90)
 except BaseException:q.kill();q.wait(timeout=10);raise
after={n:sha(r/n) for n in names};assert after==before
fixture=json.loads((out/'SUMMARY.json').read_bytes()) if (out/'SUMMARY.json').exists() else None
result=dict(actual_exit_code=code,actual_fixture_child_pid=q.pid,actual_fixture_child_reaped=q.poll() is not None,actual_fixture_child_streams_final=True,fixture=fixture,remote_fixture_summary_sha256=sha(out/'SUMMARY.json') if fixture else None,current_source_spec_sha256=after,stdout_tail=stdout.read_text(errors='replace')[-3000:],stderr_tail=stderr.read_text(errors='replace')[-3000:],new_gpu_contexts=0,production_signals_sent=0)
assert code==0 and fixture['status']=='LDM100K_CONTROL_NEW_CPU_FIXTURES_PASSED' and fixture['production_process_signals']==0 and fixture['new_gpu_contexts']==0,result
print(json.dumps(result))
'''.replace('RUN',repr(a.run))
ast.parse(code);z=json.loads(recorded_remote(code,O/('fix_'+a.run+'_client.json'),timeout=120));(O/('control_fixture_'+a.run+'.json')).write_text(json.dumps(z,indent=2),encoding='utf-8');print(json.dumps(z))

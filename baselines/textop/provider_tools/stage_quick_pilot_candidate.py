"""Stage only NEW disabled pilot files; run new CPU fixtures, never dispatch SDK."""
import ast, base64, hashlib, json, pathlib, sys, time
from session import remote
BASE=pathlib.Path(__file__).resolve().parents[1]
FILES=['scripts/quick_pilot_protocol.py','scripts/quick_pilot_trace.py','scripts/evaluate_quick_pilot.py',
       'scripts/supervise_quick_pilot.py','configs/quick-pilot-spec.json','provider_tools/validate_quick_pilot_cpu.py']
out=BASE/'results/quick_pilot_candidate_20261026';out.mkdir(exist_ok=False)
spec=json.loads((BASE/'configs/quick-pilot-spec.json').read_text(encoding='utf-8'))
assert spec['pilot_launch_enabled'] is spec['sample_launch_enabled'] is False
payload={name:base64.b64encode((BASE/name).read_bytes()).decode() for name in FILES}
for name in FILES:
 if name.endswith('.py'):ast.parse((BASE/name).read_text(encoding='utf-8'))
script='''import ast,base64,hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
assert os.getuid()==1013 and root.resolve()==root
files=PAYLOAD
plan=json.loads((root/'configs/quick-reproduction-plan-20261026.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
frozen={name:sha(root/name) for name in plan['frozen_runtime_files']}
assert frozen==plan['frozen_runtime_files']
assert all(sha(root/name)==value for name,value in plan['control_sources'].items())
assert all(not (root/name).exists() for name in files),'New candidate already staged; inspect existing evidence, never overwrite/re-run'
deployed={}
for name,encoded in files.items():
 target=root/name;target.parent.mkdir(exist_ok=True)
 assert target.parent.resolve()==target.parent and target.parent.stat().st_uid==1013
 raw=base64.b64decode(encoded)
 if name.endswith('.py'):ast.parse(raw.decode('utf-8'))
 with target.open('xb') as f:f.write(raw)
 deployed[name]=sha(target)
run=root/'tracker_validation/quick_pilot_cpu_server_v1'
assert not run.exists()
cmd=[str(root/'env/bin/python'),str(root/'provider_tools/validate_quick_pilot_cpu.py'),'--output',str(run)]
def bind():os.sched_setaffinity(0,{0,1,2,3})
cp=subprocess.run(cmd,capture_output=True,text=True,timeout=90,preexec_fn=bind)
(root/'logs/quick_pilot_cpu_server_v1.log').write_text(cp.stdout+'\\n'+cp.stderr)
summary=json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else None
assert {name:sha(root/name) for name in frozen}==frozen
assert all(sha(root/name)==value for name,value in plan['control_sources'].items())
spec=json.loads((root/'configs/quick-pilot-spec.json').read_text());assert spec['pilot_launch_enabled'] is spec['sample_launch_enabled'] is False
# Execute the new disabled supervisor to prove refusal BEFORE teacher/lock/SDK.
refuse=subprocess.run(['python3',str(root/'scripts/supervise_quick_pilot.py'),'--stage','smoke','--run','quick_pilot_disabled_preflight_v1'],capture_output=True,text=True,timeout=30)
(root/'logs/quick_pilot_disabled_preflight_v1.log').write_text(refuse.stdout+'\\n'+refuse.stderr)
result=dict(status='QUICK_PILOT_CANDIDATE_STAGED_CPU_ONLY',deployed_sha256=deployed,
 cpu_exit_code=cp.returncode,cpu_summary=summary,disabled_supervisor_exit_code=refuse.returncode,
 disabled_supervisor_stderr=refuse.stderr,disabled_no_run_directory=not (root/'tracker_validation/quick_pilot_disabled_preflight_v1').exists(),
 active16_unchanged=True,quick_control_sources_unchanged=True,pilot_launch_enabled=False,sample_launch_enabled=False,
 sdk_runs=0,training_signals_sent=0,new_optimizer_updates=0,sample_unix=time.time())
with (root/'receipts/quick_pilot_candidate_stage_v1.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
assert cp.returncode==0 and summary['check_count']==28
assert refuse.returncode!=0 and 'Quick pilot disabled' in refuse.stderr and result['disabled_no_run_directory']
'''.replace('PAYLOAD',repr(payload))
(out/'runtime_bytes.json').write_text(json.dumps(dict(deployed_files=payload,quality_label=spec['quality_label']),indent=2),encoding='utf-8')
try:
 result=json.loads(remote(script,timeout=150))
 (out/'stage.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:result[k] for k in ['status','cpu_exit_code','disabled_supervisor_exit_code','active16_unchanged','sdk_runs','training_signals_sent']}))
except BaseException as error:
 (out/'stage_failure.json').write_text(json.dumps(dict(error=repr(error),sample_unix=time.time()),indent=2),encoding='utf-8')
 raise

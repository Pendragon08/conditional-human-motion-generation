"""One bounded update to the disabled candidate after its independent review."""
import base64, json, pathlib
from session import remote
BASE=pathlib.Path(__file__).resolve().parents[1];out=BASE/'results/quick_pilot_candidate_20261026'
assert not (out/'trace_semantics_v2.json').exists()
files=['scripts/quick_pilot_trace.py','scripts/evaluate_quick_pilot.py','configs/quick-pilot-spec.json','provider_tools/validate_quick_pilot_cpu.py']
payload={f:base64.b64encode((BASE/f).read_bytes()).decode() for f in files}
script='''import ast,base64,hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
payload=PAYLOAD;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v1=json.loads((root/'receipts/quick_pilot_candidate_stage_v1.json').read_text())
assert all(sha(root/name)==value for name,value in v1['deployed_sha256'].items())
qp=json.loads((root/'configs/quick-reproduction-plan-20261026.json').read_text())
assert all(sha(root/name)==value for name,value in qp['frozen_runtime_files'].items())
assert all(sha(root/name)==value for name,value in qp['control_sources'].items())
old=json.loads((root/'configs/quick-pilot-spec.json').read_text());assert old['pilot_launch_enabled'] is old['sample_launch_enabled'] is False
snapshot=root/'receipts/quick_pilot_candidate_v1_runtime_bytes.json';assert not snapshot.exists()
prior={name:base64.b64encode((root/name).read_bytes()).decode() for name in v1['deployed_sha256']}
with snapshot.open('x') as stream:json.dump(prior,stream,indent=2)
for name,encoded in payload.items():
 raw=base64.b64decode(encoded)
 if name.endswith('.py'):ast.parse(raw.decode('utf-8'))
 target=root/name;temp=target.with_name(target.name+'.quick_pilot_v2.partial');assert not temp.exists()
 assert target.resolve()==target and target.stat().st_uid==1013
 with temp.open('xb') as stream:stream.write(raw)
 temp.replace(target)
new=json.loads((root/'configs/quick-pilot-spec.json').read_text());assert new['candidate_revision']==2 and new['pilot_launch_enabled'] is new['sample_launch_enabled'] is False
run=root/'tracker_validation/quick_pilot_cpu_server_v2';assert not run.exists()
cp=subprocess.run([str(root/'env/bin/python'),str(root/'provider_tools/validate_quick_pilot_cpu.py'),'--output',str(run)],capture_output=True,text=True,timeout=90,preexec_fn=lambda:os.sched_setaffinity(0,{0,1,2,3}))
(root/'logs/quick_pilot_cpu_server_v2.log').write_text(cp.stdout+'\\n'+cp.stderr)
result=dict(status='QUICK_PILOT_TRACE_SEMANTICS_V2_CPU_VERIFIED',prior_runtime_bytes=prior,
 deployed_sha256={name:sha(root/name) for name in v1['deployed_sha256']},cpu_exit_code=cp.returncode,
 cpu_summary=json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else None,
 active16_unchanged=all(sha(root/name)==value for name,value in qp['frozen_runtime_files'].items()),
 live_quick_controls_unchanged=all(sha(root/name)==value for name,value in qp['control_sources'].items()),
 sdk_runs=0,training_signals_sent=0,new_optimizer_updates=0,pilot_launch_enabled=False,sample_launch_enabled=False,sample_unix=time.time())
with (root/'receipts/quick_pilot_candidate_trace_v2.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result))
assert cp.returncode==0 and result['cpu_summary']['check_count']==31 and result['active16_unchanged'] and result['live_quick_controls_unchanged']
'''.replace('PAYLOAD',repr(payload))
try:
 result=json.loads(remote(script,timeout=130))
 (out/'trace_semantics_v2.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 (out/'runtime_bytes_v2.json').write_text(json.dumps(dict(updated_files=payload),indent=2),encoding='utf-8')
 print(json.dumps({k:result[k] for k in ['status','cpu_exit_code','active16_unchanged','sdk_runs','training_signals_sent']}))
except BaseException as error:
 (out/'trace_semantics_v2_failure.json').write_text(json.dumps(dict(error=repr(error)),indent=2),encoding='utf-8');raise

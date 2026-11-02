"""Deploy only NEW disabled source-selection files and run new CPU logic checks."""
import ast,base64,json,pathlib
from session import remote
B=pathlib.Path(__file__).resolve().parents[1];R=B/'results/quick_source_selection_20261026';R.mkdir(exist_ok=False)
files=['scripts/select_quick_corpus_sources.py','configs/quick-corpus-selection-spec.json','provider_tools/validate_quick_source_decisions.py']
payload={name:base64.b64encode((B/name).read_bytes()).decode() for name in files}
for name in files:
 if name.endswith('.py'):ast.parse((B/name).read_text(encoding='utf-8'))
(R/'runtime_bytes_v1.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
script='''import ast,base64,hashlib,json,os,pathlib,subprocess
root=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
files=PAYLOAD;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
qp=json.loads((root/'configs/quick-reproduction-plan-20261026.json').read_text())
bound=dict(qp['frozen_runtime_files'],**qp['control_sources'])
assert all(sha(root/name)==value for name,value in bound.items())
pilot=(root/'configs/quick-pilot-spec.json').read_bytes();genflags=json.loads(pilot)
assert genflags['pilot_launch_enabled'] is genflags['sample_launch_enabled'] is False
assert all(not (root/name).exists() for name in files),'Already staged; inspect old evidence instead of overwriting'
for name,encoded in files.items():
 target=root/name;raw=base64.b64decode(encoded)
 if name.endswith('.py'):ast.parse(raw.decode())
 assert target.parent.resolve()==target.parent and target.parent.stat().st_uid==1013
 with target.open('xb') as stream:stream.write(raw)
run=root/'tracker_validation/quick_source_decisions_cpu_v1';assert not run.exists()
cp=subprocess.run(['python3',str(root/'provider_tools/validate_quick_source_decisions.py'),'--output',str(run)],capture_output=True,text=True,timeout=30,preexec_fn=lambda:os.sched_setaffinity(0,{0,1,2,3}))
(root/'logs/quick_source_decisions_cpu_v1.log').write_text(cp.stdout+'\\n'+cp.stderr)
refuse=subprocess.run(['python3',str(root/'scripts/select_quick_corpus_sources.py'),'--spec',str(root/'configs/quick-corpus-selection-spec.json'),'--pilot-run','not_available_yet','--run','selection_disabled_v1'],capture_output=True,text=True,timeout=30)
(root/'logs/quick_source_selection_disabled_v1.log').write_text(refuse.stdout+'\\n'+refuse.stderr)
result=dict(status='QUICK_SOURCE_SELECTION_DISABLED_CANDIDATE_CPU_STAGED',cpu_exit_code=cp.returncode,
 cpu_summary=json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else None,
 disabled_exit_code=refuse.returncode,disabled_stderr=refuse.stderr,
 no_real_view_created=not (root/'quick_source_views/selection_disabled_v1').exists(),
 original16_and_controls_unchanged=all(sha(root/name)==value for name,value in bound.items()),
 pilot_spec_unchanged=(root/'configs/quick-pilot-spec.json').read_bytes()==pilot,
 deployed_sha256={name:sha(root/name) for name in files},sdk_runs=0,training_signals_sent=0,actual_source_decisions=0)
with (root/'receipts/quick_source_selection_stage_v1.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result))
assert cp.returncode==0 and result['cpu_summary']['check_count']==15
assert refuse.returncode!=0 and 'source selection disabled' in refuse.stderr
assert result['no_real_view_created'] and result['original16_and_controls_unchanged'] and result['pilot_spec_unchanged']
'''.replace('PAYLOAD',repr(payload))
try:
 result=json.loads(remote(script,timeout=90));(R/'stage.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:result[k] for k in ['status','cpu_exit_code','disabled_exit_code','original16_and_controls_unchanged','sdk_runs','actual_source_decisions']}))
except BaseException as error:
 (R/'failure.json').write_text(json.dumps(dict(error=repr(error)),indent=2),encoding='utf-8');raise

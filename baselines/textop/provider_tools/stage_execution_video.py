"""Stage the new render-only CPU profile and execute its first bounded witness."""
import base64, json, pathlib
from session import remote
B=pathlib.Path(__file__).resolve().parents[1];R=B/'results/execution_video_20261026';R.mkdir(exist_ok=False)
files=['scripts/render_quick_execution_video.py','scripts/supervise_execution_video.py','configs/execution-video-spec.json']
payload={name:base64.b64encode((B/name).read_bytes()).decode() for name in files}
(R/'runtime_bytes_v1.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
script='''import ast,base64,hashlib,json,os,pathlib,subprocess
root=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
files=PAYLOAD;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(not (root/name).exists() for name in files)
qp=json.loads((root/'configs/quick-reproduction-plan-20261026.json').read_text())
assert all(sha(root/name)==value for name,value in qp['frozen_runtime_files'].items())
assert all(sha(root/name)==value for name,value in qp['control_sources'].items())
for name,encoded in files.items():
 target=root/name;raw=base64.b64decode(encoded)
 if name.endswith('.py'):ast.parse(raw.decode())
 assert target.parent.resolve()==target.parent and target.parent.stat().st_uid==1013
 with target.open('xb') as stream:stream.write(raw)
spec=json.loads((root/'configs/execution-video-spec.json').read_text());assert spec['actual_rendering_enabled'] is False and spec['software_render_witnesses']==[]
run='renderer_author_v1';command=['python3',str(root/'scripts/supervise_execution_video.py'),'--mode','renderer_test','--run',run]
cp=subprocess.run(command,capture_output=True,text=True,timeout=170)
(root/'logs/execution_video_author_supervisor_v1.log').write_text(cp.stdout+'\\n'+cp.stderr)
receipt=root/'video_validation'/run/'receipt.json';exitpath=root/'receipts'/('execution_video_'+run+'_exit.json')
result=dict(status='EXECUTION_VIDEO_AUTHOR_WITNESS_COLLECTED',supervisor_exit_code=cp.returncode,
 source_files_sha256={name:sha(root/name) for name in files},receipt=json.loads(receipt.read_text()) if receipt.exists() else None,
 exit=json.loads(exitpath.read_text()) if exitpath.exists() else None,receipt_sha256=sha(receipt) if receipt.exists() else None,
 exit_sha256=sha(exitpath) if exitpath.exists() else None,sdk_runs=0,training_signals_sent=0)
with (root/'receipts/execution_video_author_stage_v1.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result))
'''.replace('PAYLOAD',repr(payload))
try:
 result=json.loads(remote(script,timeout=185));(R/'author_v1.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(supervisor_exit=result['supervisor_exit_code'],receipt=result.get('receipt'),exit=result.get('exit'))))
except BaseException as error:
 (R/'author_stage_failure.json').write_text(json.dumps(dict(error=repr(error)),indent=2),encoding='utf-8');raise

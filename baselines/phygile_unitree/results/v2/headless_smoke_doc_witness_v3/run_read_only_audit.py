from pathlib import Path
import importlib.util,json
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('task_ops',root/'ops.py')
ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)
preflight=json.loads(json.loads((root/'remote_preflight_result.json').read_text(encoding='utf-8'))['stdout'])
script='EXPECTED='+repr(preflight)+'\n'+(root/'remote_audit.py').read_text(encoding='utf-8')
result=ops.remote(script,python='/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/isaacsim-4.5.0/python.sh',timeout=60)
(root/'audit_transport.json').write_text(json.dumps({'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('audit_exit_code='+str(result.returncode))
if result.returncode:
 print(result.stdout);print(result.stderr)
else:
 data=json.loads(result.stdout)
 for name,blob in data.pop('source_blobs').items():
  import base64
  path=root/'remote_sources'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(blob['base64']))
 (root/'independent_audit.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(data,ensure_ascii=False,indent=2))


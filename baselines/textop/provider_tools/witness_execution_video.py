"""One documented CPU renderer witness; collect actual media hashes, never SDK."""
import base64, hashlib, json, pathlib, re
from session import remote
B=pathlib.Path(__file__).resolve().parents[1]
import argparse
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert re.fullmatch('[a-z0-9_]+',a.run)
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
files=['scripts/render_quick_execution_video.py','scripts/supervise_execution_video.py','configs/execution-video-spec.json']
expected={name:hashlib.sha256((B/name).read_bytes()).hexdigest() for name in files}
script='''import base64,hashlib,json,pathlib,subprocess
root=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');run=RUN;expected=EXPECTED
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(root/name)==value for name,value in expected.items())
spec=json.loads((root/'configs/execution-video-spec.json').read_text());assert spec['actual_rendering_enabled'] is False
assert not (root/'video_validation'/run).exists() and not (root/'receipts'/('execution_video_'+run+'_exit.json')).exists()
command=['python3',str(root/'scripts/supervise_execution_video.py'),'--mode','renderer_test','--run',run]
cp=subprocess.run(command,capture_output=True,text=True,timeout=170)
(root/'logs'/('execution_video_'+run+'_supervisor.log')).write_text(cp.stdout+'\\n'+cp.stderr)
folder=root/'video_validation'/run;rp=folder/'receipt.json';ep=root/'receipts'/('execution_video_'+run+'_exit.json')
r=json.loads(rp.read_text()) if rp.exists() else None;e=json.loads(ep.read_text()) if ep.exists() else None
result=dict(actual_supervisor_exit_code=cp.returncode,stdout=cp.stdout,stderr=cp.stderr,receipt=r,exit=e,
 receipt_sha256=sha(rp) if rp.exists() else None,exit_sha256=sha(ep) if ep.exists() else None,
 source_bytes_unchanged=all(sha(root/name)==value for name,value in expected.items()),media={})
if cp.returncode==0:
 for name in ['first_frame.png','video.mp4']:
  result['media'][name]=base64.b64encode((folder/name).read_bytes()).decode()
print(json.dumps(result))
'''.replace('RUN',repr(a.run)).replace('EXPECTED',repr(expected))
try:
 result=json.loads(remote(script,timeout=185));media=result.pop('media')
 (out/'witness.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 assert result['actual_supervisor_exit_code']==0 and result['source_bytes_unchanged'] is True
 r=result['receipt'];e=result['exit']
 assert r['status']=='SOFTWARE_RENDERER_SYNTHETIC_WITNESS_PASSED' and r['actual_policy_result_video'] is False
 assert 'llvmpipe' in r['gl_renderer'].lower() and r['nvidia_libraries_loaded']==[] and r['new_physics_steps']==r['sdk_runs']==0
 assert r['frames_encoded']==r['frames_decoded']==24 and r['fps']==50 and r['joint_name_mapping_verified'] is True
 assert r['visual_ground_count']==1 and r['visual_ground_count_checked_in_compiled_model'] is True
 assert r['visual_ground_mode']=='reused_existing_world_plane' and r['visual_world_sections']==2
 assert e['supervision_passed'] and e['original16_unchanged'] and e['quick_controls_unchanged'] and e['training_signals_sent']==0
 for name,encoded in media.items():
  raw=base64.b64decode(encoded);assert hashlib.sha256(raw).hexdigest()==r['video_sha256' if name.endswith('mp4') else 'first_frame_sha256']
  (out/name).write_bytes(raw)
 print(json.dumps(dict(status=r['status'],supervisor_exit_code=0,frames=24,fps=50,gl_renderer=r['gl_renderer'],sdk_runs=0,training_signals_sent=0)))
except BaseException as error:
 (out/'failure.json').write_text(json.dumps(dict(error=repr(error)),indent=2),encoding='utf-8');raise

"""Reuse unchanged author/fresh Mesa kernel; declare new real-input proof wrapper."""
import ast,base64,json,pathlib
from session import remote
B=pathlib.Path(__file__).resolve().parents[1];names=['scripts/render_generated_execution.py','scripts/supervise_generated_video.py']
for n in names:ast.parse((B/n).read_text(encoding='utf-8'))
payload={n:base64.b64encode((B/n).read_bytes()).decode() for n in names}
script=r'''
import ast,base64,hashlib,json,pathlib
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
out=r/'receipts/generated_video_stage_v1.json';assert not out.exists();run='generated_sdk_author_v1';fp=r/'generated_execution_private'/run/'pipeline.json';j=json.loads(fp.read_text());assert j['status']=='GENERATED_REFERENCE_PIPELINE_TRIAL_CLOSED' and j['output_writers_closed']
for stage in ['export','replay']:
 e=json.loads((r/'receipts'/('generated_pipeline_'+run+'_'+stage+'_exit.json')).read_text());assert e['child_exit_code']==0 and e['supervision_passed']
payload=PAYLOAD
for n,v in payload.items():
 p=r/n;assert not p.exists();raw=base64.b64decode(v);ast.parse(raw.decode());p.write_bytes(raw)
old=json.loads((r/'configs/execution-video-spec.json').read_text());assert sha(r/'scripts/render_quick_execution_video.py')==old['renderer_sha256']
spec=dict(old);spec.update(generated_video_enabled=True,actual_rendering_enabled=True,pipeline_run=run,pipeline_receipt_sha256=sha(fp),generated_sdk_spec_sha256=sha(r/'configs/quick-generated-sdk-spec.json'),
 wrapper_sha256=sha(r/'scripts/render_generated_execution.py'),supervisor_sha256=sha(r/'scripts/supervise_generated_video.py'),renderer_role='Recorded IsaacLab execution of real VAE2000/LDM3000 generated refs using5k teacher; CPUvisualFK only, no new dynamics or onlineclosedloop',actual_timeout_seconds=900)
keys=['required_versions','environment','width','height','fps','test_frames','visual_model_xml_sha256','mesh_files_sha256','mesa_vendor_sha256'];signature=lambda s:hashlib.sha256(json.dumps({k:s[k] for k in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest();assert signature(old)==signature(spec)
sp=r/'configs/generated-execution-video-spec.json';assert not sp.exists();sp.write_text(json.dumps(spec,indent=2)+'\n');proof={'status':'GENERATED_VIDEO_INPUT_CONTRACT_STAGED','spec_sha256':sha(sp),'spec_base64':base64.b64encode(sp.read_bytes()).decode(),'unchanged_visual_numeric_signature':signature(spec),'new_sdk_runs':0,'new_physics_steps':0,'new_optimizer_updates':0,'media_not_created_by_staging':True}
out.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''.replace('PAYLOAD',repr(payload))
j=json.loads(remote(script,timeout=45));(B/'configs/generated-execution-video-spec.json').write_bytes(base64.b64decode(j['spec_base64']));out=B/'results/generated_execution_20261026';out.mkdir(exist_ok=True);(out/'video_stage_v1.json').write_text(json.dumps(j,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in j.items() if k!='spec_base64'}))

"""Stage a distinct bounded two-worker profile; never change old closed experiments."""
import ast,base64,json,pathlib
from session import remote
B=pathlib.Path(__file__).resolve().parents[1]
names=['scripts/online_quick_protocol.py','scripts/online_quick_generator.py','scripts/online_quick_sdk.py','scripts/supervise_online_quick.py']
for n in names:ast.parse((B/n).read_text(encoding='utf-8'))
payload={n:base64.b64encode((B/n).read_bytes()).decode() for n in names}
script=r'''
import ast,base64,fcntl,hashlib,json,pathlib
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
lk=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
sp=r/'configs/quick-online-spec.json';assert not sp.exists()
gen=json.loads((r/'configs/quick-generated-reference-spec.json').read_text());sdk=json.loads((r/'configs/quick-generated-sdk-spec.json').read_text())
old=dict(gen['bound_sources'],**sdk['bound_sources'])
old.update({'configs/quick-generated-reference-spec.json':sha(r/'configs/quick-generated-reference-spec.json'),
 'configs/quick-generated-sdk-spec.json':sha(r/'configs/quick-generated-sdk-spec.json'),
 'scripts/tracker_analytic_fk.py':sha(r/'scripts/tracker_analytic_fk.py'),'configs/tracker-analytic-fk-spec.json':sha(r/'configs/tracker-analytic-fk-spec.json')})
for n,h in old.items():assert sha(r/n)==h
w=[]
for run in ['analytic_fk_author_v1','analytic_fk_document_v1']:
 p=r/'analytic_fk_validation'/run/'receipt.json';x=json.loads(p.read_text());assert x['passed'] and x['spec_sha256']==sha(r/'configs/tracker-analytic-fk-spec.json')
 w.append({'receipt':str(p.relative_to(r)),'sha256':sha(p)})
for n,v in PAYLOAD.items():
 p=r/n;assert not p.exists();raw=base64.b64decode(v);ast.parse(raw.decode());p.write_bytes(raw);old[n]=sha(p)
er=json.loads((r/'generated_execution_private/generated_sdk_author_v1/export/receipt.json').read_text());warm=json.loads((r/'configs/tracker-env-spec.json').read_text())
s={'schema':'textop.quick_online.v1','validation_enabled':True,'full_training_enabled':False,'bindings':old,'fk_witnesses':w,
 'generator_torch_fraction':.15,'sdk_torch_fraction':.50,'cpu_affinity':[0,1,2,3],'gpu':1,'only_one_sdk':True,
 'urdf':warm['robot_urdf'],'urdf_sha256':warm['robot_urdf_sha256'],'joint_names':er['joint_names'],'body_names':er['body_names'],
 'control_steps':257,'switch_control_step':90,'prefetch_low_watermark':24,'buffer_capacity':400,'sdk_timeout_seconds':600,'job_timeout_seconds':900,
 'control_period_seconds':.02,'deadline_tolerance_seconds':0,'semantic_quality_accepted':False,'final_reference_endpoint_verified':False}
sp.write_text(json.dumps(s,indent=2)+'\n');print(json.dumps({'spec_base64':base64.b64encode(sp.read_bytes()).decode(),'spec_sha256':sha(sp),'bindings':old,'fk_witnesses':w}))
'''.replace('PAYLOAD',repr(payload))
j=json.loads(remote(script,timeout=90));(B/'configs/quick-online-spec.json').write_bytes(base64.b64decode(j['spec_base64']))
o=B/'results/online_quick_20261026';o.mkdir(exist_ok=True);(o/'stage_v1.json').write_text(json.dumps(j,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in j.items() if k!='spec_base64'}))

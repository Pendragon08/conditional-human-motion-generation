import pathlib,json,hashlib,base64,sys,ast
W=pathlib.Path(__file__).resolve().parents[2];B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(B/'provider_tools'));from backup_filter_recovery_checkpoint import recorded_remote
O=B/'results/filtered15000_handoff_tools_20261027';original=json.loads((O/'recovery_bridge_deploy_v1.json').read_text(encoding='utf-8'))
n='scripts/finalize_filtered15000_data.py';p=B/n;ast.parse(p.read_text(encoding='utf-8'));source=p.read_bytes()
sp=B/'configs/filtered15000-recovery-spec.json';old=sp.read_bytes();s=json.loads(old);assert not s['prepare_authorized'] and not s['acceptance_authorized']
s['finalizer_sha256']=hashlib.sha256(source).hexdigest();data=(json.dumps(s,indent=2)+'\n').encode()
code=r'''
import pathlib,json,hashlib,base64,ast,os
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
p=r/'scripts/finalize_filtered15000_data.py';sp=r/'configs/filtered15000-recovery-spec.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()==__OLD_SOURCE__ and hashlib.sha256(sp.read_bytes()).hexdigest()==__OLD_SPEC__
s=json.loads(sp.read_text());assert not s['prepare_authorized'] and not s['acceptance_authorized']
for name,raw in [('data_finalizer_before_provenance_v3.py',p.read_bytes()),('recovery_spec_before_provenance_v3.json',sp.read_bytes())]:
 with (r/'receipts'/name).open('xb') as f:f.write(raw)
source=base64.b64decode(__SOURCE__);spec=base64.b64decode(__SPEC__);ast.parse(source.decode('utf-8'))
for path,raw in [(p,source),(sp,spec)]:
 temp=path.with_name(path.name+'.provenance_v3.partial')
 with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 temp.replace(path)
print(json.dumps(dict(status='INACTIVE_DATA_FINALIZER_BINDING_AND_EXACT_BYTES_FIXED',source_sha256=hashlib.sha256(source).hexdigest(),
 spec_sha256=hashlib.sha256(spec).hexdigest(),old_bytes_preserved=True,new_gpu_contexts=0,new_model_loads=0,actual_acceptance_runs=0)))
'''
for k,v in [('__OLD_SOURCE__',original['written'][n]),('__OLD_SPEC__',hashlib.sha256(old).hexdigest()),('__SOURCE__',base64.b64encode(source).decode()),('__SPEC__',base64.b64encode(data).decode())]:code=code.replace(k,repr(v))
ast.parse(code);r=json.loads(recorded_remote(code,O/'recovery_finalizer_fix_v3_client.json',timeout=90));sp.write_bytes(data)
(O/'recovery_finalizer_fix_v3.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r))

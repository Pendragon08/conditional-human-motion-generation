import pathlib,json,hashlib,base64,sys,ast
W=pathlib.Path(__file__).resolve().parents[2];B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(B/'provider_tools'));from backup_filter_recovery_checkpoint import recorded_remote
O=B/'results/filtered15000_handoff_tools_20261027';previous=json.loads((O/'recovery_bridge_deploy_v1.json').read_text(encoding='utf-8'))
n='scripts/filtered15000_corpus_acceptance.py';p=B/n;ast.parse(p.read_text(encoding='utf-8'));new_source=p.read_bytes()
sp=B/'configs/filtered15000-recovery-spec.json';old_spec=sp.read_bytes();s=json.loads(old_spec);assert not s['prepare_authorized'] and not s['acceptance_authorized']
s['bound_sources'][n]=hashlib.sha256(new_source).hexdigest();new_spec=(json.dumps(s,indent=2)+'\n').encode()
code=r'''
import pathlib,os,json,hashlib,base64,ast
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
p=r/'scripts/filtered15000_corpus_acceptance.py';sp=r/'configs/filtered15000-recovery-spec.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()==__OLD_SOURCE__ and hashlib.sha256(sp.read_bytes()).hexdigest()==__OLD_SPEC__
s=json.loads(sp.read_text());assert not s['prepare_authorized'] and not s['acceptance_authorized']
for name,data in [('corpus_acceptance_before_layout_v2.py',p.read_bytes()),('recovery_spec_before_layout_v2.json',sp.read_bytes())]:
 with (r/'receipts'/name).open('xb') as f:f.write(data)
source=base64.b64decode(__NEW_SOURCE__);spec=base64.b64decode(__NEW_SPEC__);ast.parse(source.decode('utf-8'))
for path,data in [(p,source),(sp,spec)]:
 temp=path.with_name(path.name+'.layout_v2.partial')
 with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 temp.replace(path)
print(json.dumps(dict(status='INACTIVE_RECOVERY_WITNESS_RECEIPT_LAYOUT_FIXED',source_sha256=hashlib.sha256(source).hexdigest(),
 spec_sha256=hashlib.sha256(spec).hexdigest(),old_bytes_preserved=True,new_gpu_contexts=0,new_model_loads=0,actual_recovery_runs=0)))
'''
for key,value in [('__OLD_SOURCE__',previous['written'][n]),('__OLD_SPEC__',hashlib.sha256(old_spec).hexdigest()),
 ('__NEW_SOURCE__',base64.b64encode(new_source).decode()),('__NEW_SPEC__',base64.b64encode(new_spec).decode())]:code=code.replace(key,repr(value))
ast.parse(code);result=json.loads(recorded_remote(code,O/'recovery_layout_fix_v2_client.json',timeout=90));sp.write_bytes(new_spec)
(O/'recovery_layout_fix_v2.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

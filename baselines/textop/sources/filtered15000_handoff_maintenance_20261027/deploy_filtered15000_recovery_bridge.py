import pathlib,json,hashlib,base64,ast,sys,shutil
W=pathlib.Path(__file__).resolve().parents[2];T=W/'work/textop_reproduction_20261024';B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(B/'provider_tools'));from backup_filter_recovery_checkpoint import recorded_remote
O=B/'results/filtered15000_handoff_tools_20261027'
names=['scripts/filtered15000_corpus_acceptance.py','scripts/prepare_filtered15000_recovery.py','scripts/supervise_filtered15000_recovery.py',
 'scripts/finalize_filtered15000_data.py','scripts/supervise_filtered15000_acceptance.py','configs/filtered15000-recovery-spec.json']
for n in names:
 if n.endswith('.py'):ast.parse((B/n).read_text(encoding='utf-8'))
code=r'''
import pathlib,json,hashlib,base64,ast,os
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
assert hashlib.sha256((r/'configs/teacher15000-direct-strict-spec.json').read_bytes()).hexdigest()=='91184fff75f8a0579a759abb418d4a0186d3e42a9a48c176e3d33facc768a1cd'
data=__PAYLOAD__;written={}
for name,value in data.items():
 p=r/name;assert not p.exists() and not p.is_symlink() and p.parent.resolve()==p.parent and p.parent.stat().st_uid==1013
 raw=base64.b64decode(value)
 if name.endswith('.py'):ast.parse(raw.decode('utf-8'))
 with p.open('xb') as f:f.write(raw)
 written[name]=hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((r/'configs/filtered15000-recovery-spec.json').read_text());assert not s['prepare_authorized'] and not s['acceptance_authorized']
print(json.dumps(dict(status='DISABLED_RECOVERY_AND_VAE_ACCEPTANCE_BRIDGE_STAGED',written=written,
 new_gpu_contexts=0,new_model_loads=0,new_recovery_jobs=0,new_private_archives=0,new_acceptance_jobs=0,
 new_training_runs=0,active_strict_runtime_modified=False)))
'''.replace('__PAYLOAD__',repr({n:base64.b64encode((B/n).read_bytes()).decode() for n in names}))
ast.parse(code)
result=json.loads(recorded_remote(code,O/'recovery_bridge_deploy_v1_client.json',timeout=90));(O/'recovery_bridge_deploy_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
M=B/'sources/filtered15000_handoff_maintenance_20261027'
for name in ['prepare_filtered15000_recovery_spec.py','deploy_filtered15000_recovery_bridge.py']:
 dst=M/name;assert not dst.exists();shutil.copy2(T/name,dst)
print(json.dumps(result))

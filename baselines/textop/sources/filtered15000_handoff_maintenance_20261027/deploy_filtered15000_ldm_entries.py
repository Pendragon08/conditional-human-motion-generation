import pathlib,json,hashlib,base64,ast,sys
W=pathlib.Path.cwd();B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop';T=W/'work/textop_reproduction_20261024'
sys.path.insert(0,str(B/'provider_tools'));from backup_filter_recovery_checkpoint import recorded_remote
O=B/'results/filtered15000_handoff_tools_20261027'
names=['scripts/filtered15000_vae_completion.py','scripts/export_filtered15000_vae.py','scripts/supervise_filtered15000_vae_export.py','scripts/run_filtered15000_ldm.py','scripts/supervise_filtered15000_ldm.py','configs/filtered15000-ldm-spec.json','configs/filtered15000-vae-export-spec.json']
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
for name in ['filtered15000-ldm-spec.json','filtered15000-vae-export-spec.json']:
 s=json.loads((r/'configs'/name).read_text())
 assert not s.get('witness_authorized',False) and not s.get('train_authorized',False) and not s.get('export_authorized',False)
 for n,h in s['bound_sources'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
plan=json.loads((r/'configs/quick-reproduction-plan-20261026.json').read_text())
for n,h in dict(plan['frozen_runtime_files'],**plan['control_sources']).items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
print(json.dumps(dict(status='DISABLED_FULL_LDM_ENTRIES_AND_CPU_EXPORT_STAGED',written=written,new_gpu_contexts=0,new_model_loads=0,new_cpu_exports=0,new_ldm_runs=0,active_strict_runtime_modified=False,old_frozen_bindings_unchanged=True)))
'''.replace('__PAYLOAD__',repr({n:base64.b64encode((B/n).read_bytes()).decode() for n in names}))
ast.parse(code)
result=json.loads(recorded_remote(code,O/'ldm_entries_deploy_v1_client.json',timeout=90));(O/'ldm_entries_deploy_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))

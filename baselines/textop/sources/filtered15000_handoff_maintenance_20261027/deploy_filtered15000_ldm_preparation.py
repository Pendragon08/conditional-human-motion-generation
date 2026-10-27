import pathlib,sys,json,hashlib,base64,ast,shutil
W=pathlib.Path(__file__).resolve().parents[2];T=W/'work/textop_reproduction_20261024';B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(B/'provider_tools'));from backup_filter_recovery_checkpoint import recorded_remote
O=B/'results/filtered15000_handoff_tools_20261027'
names=['scripts/filtered15000_ldm_runtime.py','scripts/full_ldm_history.py','configs/filtered15000-ldm-plan.json']
for n in names:
 if n.endswith('.py'):ast.parse((B/n).read_text(encoding='utf-8'))
code=r'''
import pathlib,json,hashlib,base64,os,ast
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
assert hashlib.sha256((r/'configs/teacher15000-direct-strict-spec.json').read_bytes()).hexdigest()=='91184fff75f8a0579a759abb418d4a0186d3e42a9a48c176e3d33facc768a1cd'
data=PAYLOAD;written={}
for name,value in data.items():
 p=r/name;assert not p.exists() and not p.is_symlink() and p.parent.stat().st_uid==1013 and p.parent.resolve()==p.parent
 raw=base64.b64decode(value)
 if name.endswith('.py'):ast.parse(raw.decode('utf-8'))
 with p.open('xb') as f:f.write(raw)
 written[name]=hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((r/'configs/filtered15000-ldm-plan.json').read_text());assert not s['witness_authorized'] and not s['train_authorized']
assert s['stages']==[50000,50000,50000,150000] and s['total_primitive_steps']==300000 and s['use_full_sample']
print(json.dumps(dict(status='DISABLED_FULL300K_LDM_NUMERICAL_SOURCE_STAGED',written=written,active_replay_modified=False,
 new_gpu_contexts=0,actual_model_loads=0,actual_sampling_or_gradients=0,full_entry_and_real_acceptance_pending=True)))
'''.replace('PAYLOAD',repr({n:base64.b64encode((B/n).read_bytes()).decode() for n in names}))
result=json.loads(recorded_remote(code,O/'ldm_runtime_deploy_v1_client.json',timeout=90));(O/'ldm_runtime_deploy_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
M=B/'sources/filtered15000_handoff_maintenance_20261027'
for name in ['prepare_filtered15000_ldm_runtime.py','check_filtered15000_ldm_preparation.py','deploy_filtered15000_ldm_preparation.py']:
 dst=M/name;assert not dst.exists();shutil.copy2(T/name,dst)
print(json.dumps(result))

import pathlib,json,hashlib,sys,base64,ast
W=pathlib.Path(__file__).resolve().parents[2];B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(B/'provider_tools'));from backup_filter_recovery_checkpoint import recorded_remote
O=B/'results/filtered15000_handoff_tools_20261027';O.mkdir(exist_ok=True)
names=['scripts/audit_teacher15000_full_result.py','scripts/supervise_teacher15000_full_audit.py','scripts/filtered15000_inputs.py',
 'scripts/build_filtered15000_corpus.py','scripts/supervise_filtered15000_corpus.py','scripts/run_filtered15000_vae.py','scripts/supervise_filtered15000_vae.py',
 'configs/teacher15000-full-result-audit-spec.json','configs/filtered15000-corpus-spec.json','configs/filtered15000-vae-spec.json']
for n in names:
 if n.endswith('.py'):ast.parse((B/n).read_text(encoding='utf-8'))
old=json.loads((O/'cpu_static_v1.json').read_text(encoding='utf-8'))
proof=dict(status='STAGED_HANDOFF_SOURCES_CHECKED_NO_ACTUAL_EXECUTION',prior_cpu_primitive_proof_sha256=hashlib.sha256((O/'cpu_static_v1.json').read_bytes()).hexdigest(),
 actual_closed_result_audits=0,actual_corpus_jobs=0,actual_gradient_witnesses=0,actual_generator_training_runs=0,
 current_source_sha256={n:hashlib.sha256((B/n).read_bytes()).hexdigest() for n in names},
 new_vae_witness_numeric_binding_static_check=True,positive_actual_acceptance_pending=True)
assert 'runtime_binding_sha256' in (B/'scripts/run_filtered15000_vae.py').read_text(encoding='utf-8')
(O/'static_final_v2.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
payload={n:base64.b64encode((B/n).read_bytes()).decode() for n in names}
code=r'''
import pathlib,hashlib,json,base64,ast,os,sys,shutil,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
assert hashlib.sha256((r/'configs/teacher15000-direct-strict-spec.json').read_bytes()).hexdigest()=='91184fff75f8a0579a759abb418d4a0186d3e42a9a48c176e3d33facc768a1cd'
data=PAYLOAD;written={}
for name,value in data.items():
 p=r/name;assert not p.exists() and not p.is_symlink() and p.parent.stat().st_uid==1013 and p.parent.resolve()==p.parent
 raw=base64.b64decode(value)
 if name.endswith('.py'):ast.parse(raw.decode('utf-8'))
 with p.open('xb') as f:f.write(raw)
 written[name]=hashlib.sha256(p.read_bytes()).hexdigest()
for name,key in [('teacher15000-full-result-audit-spec.json','launch_enabled'),('filtered15000-corpus-spec.json','build_authorized'),('filtered15000-vae-spec.json','witness_authorized')]:
 assert json.loads((r/'configs'/name).read_text())[key] is False
assert not json.loads((r/'configs/filtered15000-vae-spec.json').read_text())['train_authorized']
ownedhome=pathlib.Path.home();candidates=[]
for p in [r,ownedhome]:
 candidates.append(dict(path=str(p),owner_uid=p.stat().st_uid,resolve_exact=p.resolve()==p,free_bytes=shutil.disk_usage(p).free,
 filesystem=subprocess.check_output(['findmnt','-T',str(p),'-n','-o','FSTYPE'],text=True).strip(),accepted_for_future_training=False))
print(json.dumps(dict(status='NEW_HANDOFF_SOURCE_STAGED_ALL_LAUNCH_FLAGS_FALSE',written=written,python=sys.version.split()[0],
 old_production_modified=False,new_gpu_contexts=0,new_model_loads=0,new_sdk_runs=0,new_audit_jobs=0,new_corpus_jobs=0,new_generator_jobs=0,
 capacity_candidates=candidates)))
'''.replace('PAYLOAD',repr(payload))
result=json.loads(recorded_remote(code,O/'deploy_v1_client.json',timeout=90));(O/'deploy_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

import pathlib,json,hashlib,ast
W=pathlib.Path(__file__).resolve().parents[2];B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=['scripts/filtered15000_corpus_acceptance.py','scripts/prepare_filtered15000_recovery.py','scripts/supervise_filtered15000_recovery.py',
 'scripts/finalize_filtered15000_data.py','scripts/supervise_filtered15000_acceptance.py',
 'provider_tools/activate_filtered15000_recovery.py','provider_tools/backup_filtered15000_recovery.py',
 'provider_tools/sync_filtered15000_witness_activation.py','provider_tools/enable_filtered15000_vae_train.py',
 'provider_tools/launch_filtered15000_handoff.py','provider_tools/collect_filtered15000_handoff.py']
trees={n:ast.parse((B/n).read_text(encoding='utf-8')) for n in names}
rendered=[]
def render(v):
 if isinstance(v,ast.Constant) and isinstance(v.value,str):return v.value
 assert isinstance(v,ast.Call) and isinstance(v.func,ast.Attribute) and v.func.attr=='replace'
 return render(v.func.value).replace(ast.literal_eval(v.args[0]),repr('fixture'))
for n,t in trees.items():
 if not n.startswith('provider_tools/'):continue
 for statement in t.body:
  if isinstance(statement,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='code' for x in statement.targets):
   code=render(statement.value);rt=ast.parse(code);rendered.append(n)
   if n.endswith('launch_filtered15000_handoff.py'):
    driver=next(x for x in rt.body if isinstance(x,ast.Assign) and any(isinstance(y,ast.Name) and y.id=='text' for y in x.targets))
    text=eval(compile(ast.Expression(driver.value),'<new-driver-format>','eval'),{'cmd':['python3','not_executed.py'],'outer':pathlib.Path('/tmp/not_written.json')})
    ast.parse(text)
for name in names[:5]:
 for node in ast.walk(trees[name]):
  if isinstance(node,ast.Import):mods=[x.name.split('.')[0] for x in node.names]
  elif isinstance(node,ast.ImportFrom):mods=[(node.module or '').split('.')[0]]
  else:continue
  assert not set(mods)&{'torch','numpy','isaaclab','gymnasium'},(name,mods)
spec=dict(schema='textop.filtered15000_data_recovery.v1',root='/data/code/ouyanghao/oyh/textop-reproduction-20261024',
 prepare_authorized=False,acceptance_authorized=False,corpus_spec_sha256=None,corpus_witnesses=[],fresh_document_summary_sha256=None,vae_spec_sha256=None,
 recovery_run='filtered15000_recovery_v1',minimum_ram_free_bytes=16*1024**3,minimum_disk_free_bytes=3*1024**3,timeout_seconds=900,
 maximum_recovery_bytes=128*1024**2,executor_sha256=sha(B/'scripts/prepare_filtered15000_recovery.py'),
 supervisor_sha256=sha(B/'scripts/supervise_filtered15000_recovery.py'),finalizer_sha256=sha(B/'scripts/finalize_filtered15000_data.py'),
 acceptance_supervisor_sha256=sha(B/'scripts/supervise_filtered15000_acceptance.py'),
 bound_sources={n:sha(B/n) for n in ['scripts/filtered15000_corpus_acceptance.py','scripts/filtered15000_inputs.py','scripts/checkpoint_storage_guard.py','scripts/contact_process_supervision.py','scripts/contact_filter_resources.py']},
 actual_corpus_recovery_prepared=False,actual_private_D_backup_verified=False,actual_data_acceptance_completed=False)
p=B/'configs/filtered15000-recovery-spec.json';assert not p.exists();p.write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
proof=dict(status='NEW_RECOVERY_AND_VAE_ENABLE_SOURCE_RENDER_CHECKS_PASSED',python_source_count=len(names),remote_renderers_syntax_checked=rendered,
 generated_durable_driver_syntax_checked=True,CPU_sources_have_no_Torch_SDK_imports=True,
 actual_recovery_jobs=0,actual_archives=0,actual_private_D_transfers=0,actual_data_acceptance=0,actual_model_loads=0,new_gpu_contexts=0,
 note='These are source checks only. The real corpus/audit inputs do not yet exist, all launch flags remain disabled.',
 source_sha256={n:sha(B/n) for n in names})
p=B/'results/filtered15000_handoff_tools_20261027/recovery_bridge_source_v1.json';assert not p.exists();p.write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof))

import base64,hashlib,json,pathlib,shutil,sys
TASK=pathlib.Path(__file__).resolve().parent;BASE=TASK.parent/'phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(BASE/'provider_tools'));from session import remote
names=['scripts/pause_generator8278_with_marker.py','scripts/audit_generator8278_pause_cpu.py']
payload={n:base64.b64encode((BASE/n).read_bytes()).decode() for n in names}
code=r'''import base64,hashlib,json,os,pathlib,shutil,subprocess,sys
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');sys.path.insert(0,str(ROOT/'scripts'))
from closed_boundary_interrupt import identity
payload=PAYLOAD
snapshot=ROOT/'pause_marker_fixtures/production_prepare_v1';snapshot.mkdir(exist_ok=False)
for name,enc in payload.items():
 data=base64.b64decode(enc);compile(data,name,'exec');p=ROOT/name
 if p.exists():(snapshot/(p.name+'.before')).write_bytes(p.read_bytes())
 p.write_bytes(data)
run='generator8278_vae200k_train_v1';matches=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=1013:continue
  ident=identity(int(p.name))
  if ident['argv']==[str(ROOT/'env/bin/python'),str(ROOT/'scripts/run_generator8278_vae.py'),'--mode','train','--run',run]:matches.append(ident)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
assert len(matches)==1,matches
names=['scripts/run_generator8278_vae.py','scripts/generator8278_vae_runtime.py','scripts/supervise_generator8278_vae.py','scripts/generator_training_state.py','scripts/checkpoint_storage_guard.py','configs/generator8278-vae-spec.json','scripts/closed_boundary_interrupt.py','scripts/quick_pidfd.py','scripts/closed_boundary_marker.py','scripts/validate_closed_boundary_marker.py','scripts/audit_generator8278_pause_cpu.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lines=(ROOT/'scripts/run_generator8278_vae.py').read_text().splitlines()
boundary=[i+1 for i,x in enumerate(lines) if "save();print(json.dumps({'step':step," in x];assert len(boundary)==1
s=dict(schema='textop.generator8278_owned_marker_pause.v1',user_request='User2026-10-27: safely save and pause this prefilter generator; complete full-library physical filtering before filtered generator training',
 run=run,handoff_name='generator8278_to_filtered_order_v1',pause_authorized=False,process_identity=matches[0],frozen_sources={n:sha(ROOT/n) for n in names},
 entrypoint_sha256=sha(ROOT/'scripts/pause_generator8278_with_marker.py'),normal_boundary_save_lineno=boundary[0],cpu_witnesses=[])
sp=ROOT/'configs/generator8278-marker-pause-spec.json';assert not sp.exists();sp.write_text(json.dumps(s,indent=2))
r=subprocess.run(['taskset','-c','0-3','python3',str(ROOT/'scripts/pause_generator8278_with_marker.py')],capture_output=True,text=True,timeout=30)
assert r.returncode==0,r.stderr
rp=ROOT/'generator8278_20261027/vae_runs'/run/'receipt.json'
print(json.dumps(dict(spec=s,preflight=json.loads(r.stdout),original_receipt=json.loads(rp.read_text()),free={str(p):shutil.disk_usage(p).free for p in [ROOT,pathlib.Path('/home/ouyanghao')]},
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader'],text=True),memory=pathlib.Path('/proc/meminfo').read_text())))
'''.replace('PAYLOAD',repr(payload));compile(code,'remote','exec')
j=json.loads(remote(code,timeout=60));out=BASE/'results/generator_pause_20261027'
(out/'marker_production_preflight_v1.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
(BASE/'configs/generator8278-marker-pause-spec.json').write_text(json.dumps(j['spec'],indent=2),encoding='utf-8')
print(json.dumps({'preflight':j['preflight'],'free':j['free'],'local_D_free':shutil.disk_usage('D:/').free,'step':j['original_receipt']['completed_primitive_attempts']}))

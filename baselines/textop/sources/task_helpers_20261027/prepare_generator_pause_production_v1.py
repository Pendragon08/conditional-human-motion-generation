import base64,hashlib,json,pathlib,shutil,sys
TASK=pathlib.Path(__file__).resolve().parent;BASE=TASK.parent/'phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(BASE/'provider_tools'));from session import remote
name='scripts/pause_generator8278_at_boundary.py';enc=base64.b64encode((BASE/name).read_bytes()).decode()
code='''import base64,hashlib,json,os,pathlib,shutil,subprocess,sys
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');sys.path.insert(0,str(ROOT/'scripts'))
from closed_boundary_interrupt import identity
data=base64.b64decode(ENC);compile(data,'pauseentry','exec');p=ROOT/'scripts/pause_generator8278_at_boundary.py';assert not p.exists();p.write_bytes(data)
run='generator8278_vae200k_train_v1';matches=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=1013:continue
  ident=identity(int(p.name))
  if ident['argv']==[str(ROOT/'env/bin/python'),str(ROOT/'scripts/run_generator8278_vae.py'),'--mode','train','--run',run]:matches.append(ident)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
assert len(matches)==1,matches
names=['scripts/run_generator8278_vae.py','scripts/generator8278_vae_runtime.py','scripts/supervise_generator8278_vae.py','scripts/generator_training_state.py','scripts/checkpoint_storage_guard.py','configs/generator8278-vae-spec.json','scripts/closed_boundary_interrupt.py','scripts/validate_closed_boundary_interrupt.py','scripts/quick_pidfd.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=dict(schema='textop.generator8278_user_pause.v1',user_request='User2026-10-27: safely save and pause this prefilter generator; complete full-library physical filtering before filtered generator training',
 run=run,handoff_name='generator8278_to_filtered_order_v1',pause_authorized=False,process_identity=matches[0],frozen_sources={n:sha(ROOT/n) for n in names},
 entrypoint_sha256=sha(ROOT/'scripts/pause_generator8278_at_boundary.py'),findmnt_executable=str(pathlib.Path(shutil.which('findmnt')).resolve()),cpu_witnesses=[])
sp=ROOT/'configs/generator8278-pause-spec.json';assert not sp.exists();sp.write_text(json.dumps(s,indent=2))
r=subprocess.run(['taskset','-c','0-3','python3',str(ROOT/'scripts/pause_generator8278_at_boundary.py')],capture_output=True,text=True,timeout=30)
assert r.returncode==0,r.stderr
driver=ROOT/'generator8278_20261027/drivers'/ (run+'.py')
print(json.dumps(dict(spec=s,preflight=json.loads(r.stdout),driver_source=driver.read_text(),free={str(p):shutil.disk_usage(p).free for p in [ROOT,pathlib.Path('/home/ouyanghao')]},
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader'],text=True))))
'''.replace('ENC',repr(enc));compile(code,'remote','exec')
j=json.loads(remote(code,timeout=60));out=BASE/'results/generator_pause_20261027'
(out/'production_preflight_v1.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
(BASE/'configs/generator8278-pause-spec.json').write_text(json.dumps(j['spec'],indent=2),encoding='utf-8')
print(json.dumps({'preflight':j['preflight'],'free':j['free'],'local_D_free':shutil.disk_usage('D:/').free,'driver_source':j['driver_source']}))

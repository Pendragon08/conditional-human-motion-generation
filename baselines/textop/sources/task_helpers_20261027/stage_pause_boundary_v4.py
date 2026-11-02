import base64, hashlib, json, pathlib, sys
TASK=pathlib.Path(__file__).resolve().parent
BASE=TASK.parent/'phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(BASE/'provider_tools'))
from session import remote
names=['scripts/closed_boundary_interrupt.py','scripts/validate_closed_boundary_interrupt.py']
payload={n:base64.b64encode((BASE/n).read_bytes()).decode() for n in names}
code='''import base64,hashlib,json,os,pathlib,subprocess,sys
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
assert os.getuid()==1013
payload=PAYLOAD
for name,enc in payload.items():
 data=base64.b64decode(enc);compile(data,name,'exec');p=ROOT/name
 assert p.exists(),str(p)
 old=ROOT/'pause_control_fixtures/source_before_v4';old.mkdir(parents=True,exist_ok=True)
 (old/pathlib.Path(name).name).write_bytes(p.read_bytes())
for name,enc in payload.items():(ROOT/name).write_bytes(base64.b64decode(enc))
before={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['scripts/run_generator8278_vae.py','scripts/generator8278_vae_runtime.py','scripts/supervise_generator8278_vae.py','configs/generator8278-vae-spec.json']}
cmd=['taskset','-c','0-3','python3',str(ROOT/'scripts/validate_closed_boundary_interrupt.py'),'--run','boundary_author_v4']
r=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
assert r.returncode==0,r.stderr
result=json.loads(r.stdout)
after={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in before};assert before==after
print(json.dumps(dict(result=result,active_bindings_unchanged=before,stderr=r.stderr)))
'''.replace('PAYLOAD',repr(payload))
compile(code,'remote_stage_pause_boundary_v1','exec')
out=BASE/'results/generator_pause_20261027';out.mkdir(exist_ok=True)
try:
 result=json.loads(remote(code,timeout=90));(out/'cpu_author_v4.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result))
except BaseException as e:
 (out/'cpu_author_v4_failure.json').write_text(json.dumps({'error':repr(e)}),encoding='utf-8');raise

import pathlib,json,hashlib,sys,base64
W=pathlib.Path(__file__).resolve().parents[2];B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(B/'provider_tools'));from backup_filter_recovery_checkpoint import recorded_remote
names=['scripts/direct_strict_trace.py','scripts/teacher15000_strict_protocol.py','scripts/evaluate_teacher15000_strict.py','scripts/supervise_teacher15000_strict.py','configs/teacher15000-direct-strict-spec.json']
payload={n:base64.b64encode((B/n).read_bytes()).decode() for n in names}
code="""import pathlib,json,base64,hashlib,os,sys,shutil,subprocess,fcntl
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
data=PAYLOAD;written={}
for n,v in data.items():
 p=r/n;assert not p.exists() and not p.is_symlink() and p.parent.stat().st_uid==1013
 with p.open('xb') as f:f.write(base64.b64decode(v))
 written[n]=hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0,str(r/'scripts'))
from teacher15000_strict_protocol import load_teacher,authorize_teacher,settings_signature
s=json.loads((r/'configs/teacher15000-direct-strict-spec.json').read_text());t=load_teacher(r,s);authorize_teacher(s,t)
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
mem=[int(x.split()[1])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0]
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
print(json.dumps(dict(written=written,teacher_stopped_verified=True,common_lock_free=True,data_free=shutil.disk_usage(r).free,mem_available=mem,gpu=gpu,settings_signature=settings_signature(s))))
""".replace('PAYLOAD',repr(payload))
o=B/'results/teacher15000_direct_strict_20261027';o.mkdir(parents=True,exist_ok=True)
result=json.loads(recorded_remote(code,o/'deploy_v1_client.json',timeout=90));(o/'deploy_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

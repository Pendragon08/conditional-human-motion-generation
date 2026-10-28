"""Deploy only brand-new disabled100k successors; freeze active VAE and controller."""
import argparse,ast,base64,hashlib,json,pathlib
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1];O=B/'results/vae100k_user_handoff_20261028'
names=['scripts/user_vae100k_completion.py','scripts/validate_user_vae100k.py','scripts/supervise_user_vae100k_quality.py','scripts/export_user_vae100k.py','scripts/supervise_user_vae100k_export.py','scripts/run_user_vae100k_ldm.py','scripts/supervise_user_vae100k_ldm.py','configs/user-vae100k-quality-spec.json','configs/user-vae100k-export-spec.json','configs/user-vae100k-ldm-spec.json']
p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);a=p.parse_args();assert a.evidence.isascii() and a.evidence.isidentifier()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n in names:
 if n.endswith('.py'):ast.parse((B/n).read_text(encoding='utf-8'),feature_version=(3,8))
specs=[json.loads((B/n).read_bytes()) for n in names if n.endswith('.json')]
for s in specs:
 assert not any(s.get(k,False) for k in ['quality_authorized','export_authorized','witness_authorized','train_authorized']) and s['vae_training'] is None
 for n,h in s['bound_sources'].items():assert sha(B/n)==h,n
payload={n:base64.b64encode((B/n).read_bytes()).decode() for n in names}
frozen={n:sha(B/n) for n in ['configs/filtered15000-vae-spec.json','scripts/run_filtered15000_vae.py','scripts/generator8278_vae_runtime.py','configs/vae100k-user-control-spec.json','scripts/vae100k_control_protocol.py','scripts/watch_vae100k_handoff.py','scripts/supervise_vae100k_handoff.py']}
code=r'''
import ast,base64,hashlib,json,os,pathlib,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
frozen=__FROZEN__;items=__ITEMS__;specs=__SPECS__
for n,h in frozen.items():assert sha(r/n)==h,n
for n,b in items.items():
 p=r/n;assert not p.exists() and not p.is_symlink() and p.parent.resolve()==p.parent and p.parent.stat().st_uid==1013,n
 raw=base64.b64decode(b)
 if n.endswith('.py'):ast.parse(raw,filename=n)
assert shutil.disk_usage(r).free>2*1024**3
for n,b in items.items():
 with (r/n).open('xb') as f:f.write(base64.b64decode(b));f.flush();os.fsync(f.fileno())
for s in specs:
 assert not any(s.get(k,False) for k in ['quality_authorized','export_authorized','witness_authorized','train_authorized'])
 for n,h in s['bound_sources'].items():assert sha(r/n)==h,n
for n,h in frozen.items():assert sha(r/n)==h,n
print(json.dumps(dict(status='USER_VAE100K_SUCCESSOR_SOURCES_DEPLOYED_DISABLED_NOT_ACCEPTED',files_sha256={n:sha(r/n) for n in items},frozen_sha256=frozen,new_gpu_contexts=0,new_model_loads=0,signals_sent=0)))
'''
for k,v in [('__FROZEN__',frozen),('__ITEMS__',payload),('__SPECS__',specs)]:code=code.replace(k,repr(v))
ast.parse(code);result=json.loads(recorded_remote(code,O/(a.evidence+'_client.json'),timeout=120))
(O/(a.evidence+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

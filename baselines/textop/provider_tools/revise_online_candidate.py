"""Explicit new candidate only after previous owned joint run actually closed."""
import argparse,base64,json,pathlib
from session import remote
p=argparse.ArgumentParser();p.add_argument('--previous-run',required=True);p.add_argument('--revision',required=True);a=p.parse_args()
assert a.previous_run.isidentifier() and a.revision.isidentifier()
B=pathlib.Path(__file__).resolve().parents[1];names=['scripts/online_quick_protocol.py','scripts/online_quick_generator.py','scripts/online_quick_sdk.py','scripts/supervise_online_quick.py']
payload={n:base64.b64encode((B/n).read_bytes()).decode() for n in names}
code=r'''
import ast,pathlib,json,base64,hashlib,fcntl
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
f=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
ex=json.loads((r/'receipts'/('online_'+PREVIOUS+'_outer_exit.json')).read_text());assert 'actual_returncode' in ex
proof=json.loads((r/'online_private'/PREVIOUS/'pipeline.json').read_text());assert proof.get('actual_generator_returncode') is not None
sp=r/'configs/quick-online-spec.json';s=json.loads(sp.read_text());paths=list(PAYLOAD)+['configs/quick-online-spec.json']
old={n:{'sha256':sha(r/n),'base64':base64.b64encode((r/n).read_bytes()).decode()} for n in paths}
dest=r/'receipts'/('online_revision_'+REVISION+'.json');assert not dest.exists()
for n,v in PAYLOAD.items():
 raw=base64.b64decode(v);ast.parse(raw.decode());(r/n).write_bytes(raw);s['bindings'][n]=sha(r/n)
s['profile_revision']=REVISION;sp.write_text(json.dumps(s,indent=2)+'\n')
res={'previous_run':PREVIOUS,'actual_previous_returncode':ex['actual_returncode'],'original_bytes':old,'new_spec_sha256':sha(sp),'new_bindings':s['bindings']}
dest.write_text(json.dumps(res,indent=2));print(json.dumps({'evidence':res,'spec_base64':base64.b64encode(sp.read_bytes()).decode()}))
'''.replace('PREVIOUS',repr(a.previous_run)).replace('REVISION',repr(a.revision)).replace('PAYLOAD',repr(payload))
j=json.loads(remote(code));(B/'configs/quick-online-spec.json').write_bytes(base64.b64decode(j['spec_base64']))
(B/'results/online_quick_20261026'/('revision_'+a.revision+'.json')).write_text(json.dumps(j['evidence'],indent=2),encoding='utf-8');print(j['evidence']['new_spec_sha256'])

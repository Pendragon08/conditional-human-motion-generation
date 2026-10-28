"""Strengthen the still-disabled new exporter; preserve all prior source bytes."""
import ast
import base64
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
O = B / 'results/ldm100k_user_handoff_20261028'
oldroot = B / 'sources/ldm100k_prefix_export_disabled_v1'
sp = B / 'configs/ldm100k-prefix-export-spec.json'
old = sp.read_bytes()
assert old == (oldroot / 'configs/ldm100k-prefix-export-spec.json').read_bytes()
s = json.loads(old)
assert not s['export_authorized'] and s['ldm_training'] is None
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
names = ['scripts/ldm100k_prefix_completion.py', 'provider_tools/test_ldm100k_prefix_metadata.py']
for n in names:
    ast.parse((B / n).read_bytes(), feature_version=(3, 8))
s['bound_sources']['scripts/ldm100k_prefix_completion.py'] = sha(B / names[0])
s['preparation_revision'] = 2
raw = (json.dumps(s, indent=2, ensure_ascii=False) + '\n').encode()
payload = {n: base64.b64encode((B / n).read_bytes()).decode() for n in names}
payload['configs/ldm100k-prefix-export-spec.json'] = base64.b64encode(raw).decode()
previous = {n: sha(oldroot / n) for n in payload}
code = r'''
import ast,base64,hashlib,json,os,pathlib,subprocess,sys
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();previous=PREVIOUS;payload=PAYLOAD;s=POLICY
current=json.loads((r/'configs/ldm100k-prefix-export-spec.json').read_bytes());assert not current['export_authorized'] and current['ldm_training'] is None
for n,h in previous.items():assert sha(r/n)==h,n
for n,h in s['bound_sources'].items():
 if n not in payload:assert sha(r/n)==h,n
archive=r/'receipts/ldm100k_prefix_export_disabled_v1';archive.mkdir(exist_ok=False)
for n,b in payload.items():
 p=archive/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write((r/n).read_bytes());f.flush();os.fsync(f.fileno())
 assert sha(p)==previous[n]
 raw=base64.b64decode(b)
 if n.endswith('.py'):ast.parse(raw,filename=n)
for n,b in payload.items():
 p=r/n;temp=p.with_name(p.name+'.disabled_v2.partial')
 with temp.open('xb') as f:f.write(base64.b64decode(b));f.flush();os.fsync(f.fileno())
 temp.replace(p)
out=r/'ldm100k_user_handoff_20261028/prefix_metadata_server_v2';assert not out.exists()
child=subprocess.Popen([sys.executable,str(r/'provider_tools/test_ldm100k_prefix_metadata.py'),'--output',str(out)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
stdout,stderr=child.communicate(timeout=60);assert child.returncode==0,stderr.decode()
fixture=json.loads((out/'SUMMARY.json').read_bytes());assert fixture['rejected_cases']==26
for n,h in s['bound_sources'].items():assert sha(r/n)==h,n
print(json.dumps(dict(status='DISABLED_PREFIX_EXPORT_INTERRUPTED_EVIDENCE_HASHES_STRENGTHENED',
 previous_sources_preserved_sha256=previous,files_sha256={n:sha(r/n) for n in payload},
 fixture=fixture,actual_fixture_exit_code=child.returncode,actual_fixture_reaped=True,actual_fixture_streams_final=True,
 actual_production_checkpoint_reads=0,actual_model_loads=0,actual_exports=0,actual_gpu_contexts=0,
 active_training_and_control_source_modified=False,export_authorized=False)))
'''.replace('PREVIOUS', repr(previous)).replace('PAYLOAD', repr(payload)).replace('POLICY', repr(s))
ast.parse(code, feature_version=(3, 8))
z = json.loads(recorded_remote(code, O / 'prefix_fix_client.json', timeout=90))
sp.write_bytes(raw)
(O / 'prefix_export_preparation_v2.json').write_text(json.dumps(z, indent=2), encoding='utf-8')
print(json.dumps(z))

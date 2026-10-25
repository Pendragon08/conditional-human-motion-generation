"""Collect aggregate evidence and privately back up numerical outlier snapshots."""
import argparse
import base64
import hashlib
import json
import pathlib
import re
import shutil
from session import remote

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
a = p.parse_args()
assert re.fullmatch('[a-z0-9_]+', a.run)
base = pathlib.Path(__file__).resolve().parents[1]
public = base / 'results/full_library_numerical_20261026' / a.run
private = pathlib.Path('D:/具身智能/TextOp复现_20261024/datasets/numerical_diagnostics_private_20261026') / a.run
public.mkdir(parents=True, exist_ok=True)
private.mkdir(parents=True, exist_ok=True)
assert shutil.disk_usage(private).free > 64 * 1024**2
script = r'''
import base64,hashlib,json,pathlib
root=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
run=RUN
out=root/'tracker_validation'/run
proof=root/'receipts'/('full_library_numerical_'+run+'_exit.json')
assert proof.exists(), 'Wait for actual child exit; do not collect a running trace'
result={'public':{},'private':{}}
paths={'exit.json':proof,'receipt.json':out/'receipt.json',
       'numerical_summary.json':out/'numerical_summary.json',
       'numerical_trace.jsonl':out/'numerical_trace.jsonl'}
for name,path in paths.items():
 if path.exists():result['public'][name]=path.read_text()
raw=[root/'logs'/('full_library_numerical_'+run+'.log')]
if (out/'numerical_private').exists():raw+=list((out/'numerical_private').iterdir())
for path in raw:
 assert path.is_file() and path.stat().st_size<32*1024**2
 data=path.read_bytes()
 result['private'][path.name]={'sha256':hashlib.sha256(data).hexdigest(),'size':len(data),'base64':base64.b64encode(data).decode()}
print(json.dumps(result))
'''.replace('RUN', repr(a.run))
result = json.loads(remote(script, timeout=180))
for name, value in result['public'].items():
    (public / name).write_text(value, encoding='utf-8')
manifest = {}
for name, value in result['private'].items():
    data = base64.b64decode(value['base64'])
    assert hashlib.sha256(data).hexdigest() == value['sha256']
    path = private / name
    assert not path.exists() or path.read_bytes() == data
    path.write_bytes(data)
    manifest[name] = {k: value[k] for k in ['sha256', 'size']}
(private / 'backup_receipt.json').write_text(json.dumps(manifest, indent=2))
(public / 'private_evidence_backup.json').write_text(json.dumps({
    'files': len(manifest), 'total_bytes': sum(v['size'] for v in manifest.values()),
    'all_sha256_verified': True, 'private_arrays_published': False}, indent=2))
print(json.dumps({'run': a.run, 'public_files': len(result['public']),
                  'private_backup_files': len(manifest),
                  'child_exit_code': json.loads(result['public']['exit.json'])['child_exit_code']}))

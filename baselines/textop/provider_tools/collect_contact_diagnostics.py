"""Publish only aggregates; preserve private replay arrays on the user's drive."""
import argparse, base64, hashlib, json, pathlib, re, shutil
from session import remote

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--family', required=True, choices=['contact_replay', 'full_contact'])
a = p.parse_args()
assert re.fullmatch('[a-z0-9_]+', a.run)
base = pathlib.Path(__file__).resolve().parents[1]
public = base / 'results' / (a.family + '_20261026') / a.run
private = pathlib.Path('D:/具身智能/TextOp复现_20261024/datasets/contact_diagnostics_private_20261026') / a.run
public.mkdir(parents=True, exist_ok=True)
private.mkdir(parents=True, exist_ok=True)
assert shutil.disk_usage(private).free > 128 * 1024**2
script = r'''
import base64,hashlib,json,pathlib
root=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
run=RUN
family=FAMILY
out=root/'tracker_validation'/run
proof=root/'receipts'/(family+'_'+run+'_exit.json')
assert proof.exists(), 'Actual exit evidence required'
result={'public':{},'private':{}}
paths={'exit.json':proof,'receipt.json':out/'receipt.json',
       'numerical_summary.json':out/'numerical_summary.json',
       'numerical_trace.jsonl':out/'numerical_trace.jsonl'}
for name,path in paths.items():
 if path.exists():result['public'][name]=path.read_text()
raw=[root/'logs'/(family+'_'+run+'.log')]
for folder in ['replay_private','full_contact_private','numerical_private']:
 if (out/folder).exists():raw.extend((out/folder).iterdir())
for path in raw:
 assert path.is_file() and not path.is_symlink() and path.stat().st_size<64*1024**2
 data=path.read_bytes();name=path.parent.name+'/'+path.name
 result['private'][name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'base64':base64.b64encode(data).decode()}
print(json.dumps(result))
'''.replace('RUN', repr(a.run)).replace('FAMILY', repr(a.family))
result = json.loads(remote(script, timeout=180))
for name, data in result['public'].items():
    (public / name).write_text(data, encoding='utf-8')
manifest = {}
for name, row in result['private'].items():
    data = base64.b64decode(row['base64'])
    assert hashlib.sha256(data).hexdigest() == row['sha256']
    path = private / name
    assert not path.exists() or path.read_bytes() == data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    manifest[name] = {k: row[k] for k in ['sha256', 'bytes']}
(private / 'backup_receipt.json').write_text(json.dumps(manifest, indent=2))
(public / 'private_backup.json').write_text(json.dumps({
    'files': len(manifest), 'total_bytes': sum(x['bytes'] for x in manifest.values()),
    'all_sha256_verified': True, 'private_arrays_published': False}, indent=2))
print(json.dumps({'family': a.family, 'run': a.run,
                  'private_backup_files': len(manifest),
                  'child_exit_code': json.loads(result['public']['exit.json'])['child_exit_code']}))

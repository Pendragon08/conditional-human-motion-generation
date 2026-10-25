"""Restore exact declared code/config bytes into a NEW local folder only."""
import argparse,base64,hashlib,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--destination',required=True);a=p.parse_args()
B=pathlib.Path(__file__).resolve().parents[1]
bundle=json.loads((B/'results/full_contact_20261026/formal_launch/bound_runtime_bytes.json').read_text(encoding='utf-8'))
out=pathlib.Path(a.destination).absolute();assert not out.exists();out.mkdir(parents=False)
assert out.resolve()==out and not out.is_symlink()
checked={}
for name,row in bundle['files'].items():
 rel=pathlib.PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
 path=out.joinpath(*rel.parts);assert out in path.resolve().parents
 data=base64.b64decode(row['base64']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'];checked[name]=row['sha256']
(out/'restore_receipt.json').write_text(json.dumps({'sha256_verified':checked,'server_files_modified':False,'training_started':False},indent=2))
print(json.dumps({'restored_files':len(checked),'destination':str(out),'all_byte_sha256_match':True}))

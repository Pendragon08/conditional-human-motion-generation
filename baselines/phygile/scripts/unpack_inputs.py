"""Safely unpack the published pilot inputs and verify their SHA-256 manifests."""
import argparse, hashlib, json, pathlib, tarfile
p=argparse.ArgumentParser();p.add_argument('archive',type=pathlib.Path);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[1]
if (root/'data').exists():raise FileExistsError('data already exists; use a fresh workspace')
with tarfile.open(a.archive,'r:gz') as tar:
 for member in tar.getmembers():
  if not member.isfile():continue
  path=pathlib.PurePosixPath(member.name)
  if path.is_absolute() or '..' in path.parts:raise ValueError(member.name)
  if path.parts[0] not in ['data','input_notices']:raise ValueError(member.name)
  dest=root/path;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():raise FileExistsError(dest)
  dest.write_bytes(tar.extractfile(member).read())
for split in ['train','test']:
 folder=root/'data'/split
 m=json.loads((folder/'manifest.json').read_text())
 for rec in m['records']:
  f=folder/rec['output'];assert hashlib.sha256(f.read_bytes()).hexdigest()==rec['sha256'],f
print('Pilot input hashes verified. These inputs are not the PhyGile paper dataset.')

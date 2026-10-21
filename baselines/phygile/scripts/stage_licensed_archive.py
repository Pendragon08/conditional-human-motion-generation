"""Stage a locally verified archive on the user's existing server without overwriting."""
import base64
import json
import shlex
import subprocess
from pathlib import Path


def stage(archive, verification, *, host, identity, remote_root, remote_python, output):
    subset = verification['subset']
    assert subset.replace('_', '').isalnum()
    destination = remote_root.rstrip('/') + '/' + subset
    remote_archive = destination + '/' + Path(archive).name
    needed = verification['archive_bytes'] + verification['unpacked_file_bytes'] + 5 * 1024**3
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-i', str(identity), host]

    def remote(code):
        wrapped = 'import base64;exec(base64.b64decode(' + repr(base64.b64encode(code.encode()).decode()) + '))'
        result = subprocess.run(ssh + [shlex.quote(remote_python) + ' -c ' + shlex.quote(wrapped)],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=1800)
        if result.returncode:
            raise RuntimeError('Server staging exit ' + str(result.returncode) + ': ' + result.stderr[-1800:])
        return json.loads(result.stdout)

    preflight = remote('import json,shutil\nfrom pathlib import Path\n'
        + 'p=Path(' + repr(destination) + ')\n'
        + 'base=Path("/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass").resolve()\n'
        + 'assert p.resolve().is_relative_to(base)\n'
        + 'free=shutil.disk_usage(base).free\n'
        + 'print(json.dumps({"exists":p.exists(),"free_bytes":free}))\n')
    if preflight['exists']:
        return {'subset': subset, 'status': 'SERVER_DESTINATION_ALREADY_EXISTS_REVIEW_REQUIRED', **preflight}
    if preflight['free_bytes'] < needed:
        return {'subset': subset, 'status': 'SERVER_STORAGE_INSUFFICIENT',
                'required_free_bytes_including_5gib_reserve': needed, **preflight}
    remote('import json\nfrom pathlib import Path\np=Path(' + repr(destination)
           + ')\np.mkdir(parents=True,exist_ok=False,mode=0o700)\nprint(json.dumps({"created":True}))')
    transfer = subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                               '-i', str(identity), str(archive), host + ':' + remote_archive],
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=3600)
    if transfer.returncode:
        raise RuntimeError('Archive transfer exit ' + str(transfer.returncode)
                           + '; partial server directory retained: ' + transfer.stderr[-800:])
    code = '''import hashlib,json,shutil,tarfile,zipfile
from pathlib import Path,PurePosixPath
archive=Path(ARCHIVE)
root=archive.parent/'extracted'
digest=hashlib.sha256()
with archive.open('rb') as f:
 for b in iter(lambda:f.read(4194304),b''):digest.update(b)
assert digest.hexdigest()==EXPECTED_SHA
assert archive.stat().st_size==EXPECTED_BYTES
root.mkdir(exist_ok=False)
records=[]
def target(name):
 p=PurePosixPath(name)
 assert not p.is_absolute() and '..' not in p.parts and '\\\\' not in name
 t=root.joinpath(*p.parts)
 assert t.resolve().is_relative_to(root.resolve())
 t.parent.mkdir(parents=True,exist_ok=True)
 return t
def copy(source,name,size):
 t=target(name);h=hashlib.sha256();n=0
 with t.open('xb') as f:
  while True:
   chunk=source.read(4194304)
   if not chunk:break
   f.write(chunk);h.update(chunk);n+=len(chunk)
 assert n==size
 records.append({'path':name,'bytes':n,'sha256':h.hexdigest()})
if archive.name.endswith('.zip'):
 with zipfile.ZipFile(archive) as z:
  for m in z.infolist():
   assert (m.external_attr>>16)&0o170000 != 0o120000
   if not m.is_dir():
    with z.open(m) as f:copy(f,m.filename,m.file_size)
else:
 with tarfile.open(archive,mode='r|bz2') as tar:
  for m in tar:
   assert m.isfile() or m.isdir()
   if m.isfile():
    with tar.extractfile(m) as f:copy(f,m.name,m.size)
assert sum(x['bytes'] for x in records)==EXPECTED_UNPACKED
result={'subset':SUBSET,'status':'SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED',
 'archive_sha256':digest.hexdigest(),'archive_bytes':archive.stat().st_size,
 'extracted_file_bytes':sum(x['bytes'] for x in records),'server_directory':str(archive.parent),
 'files':records,'motion_arrays_validated':False,'g1_pairs_generated':0,'new_262d_sequences':0,
 'model_forward_validated':False,'generator_training_started':False}
(archive.parent/'server_verification.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps({'subset':SUBSET,'status':result['status'],'files':len(records)}))
'''
    values = {'ARCHIVE': remote_archive, 'EXPECTED_SHA': verification['sha256'],
              'EXPECTED_BYTES': verification['archive_bytes'],
              'EXPECTED_UNPACKED': verification['unpacked_file_bytes'], 'SUBSET': subset}
    header = '\n'.join(k + '=' + repr(v) for k, v in values.items()) + '\n'
    remote(header + code)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    fetched = subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                             '-i', str(identity), host + ':' + destination + '/server_verification.json', str(output)],
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    if fetched.returncode:
        raise RuntimeError('Server verified, but receipt retrieval failed: ' + fetched.stderr[-800:])
    result = json.loads(output.read_text())
    return {k: v for k, v in result.items() if k != 'files'}

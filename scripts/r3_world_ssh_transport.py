"""Bounded SSH with argument-safe UTF-8/base64 transport; no automatic retries."""
from pathlib import Path
import base64
import json
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/data/code/ouyanghao/oyh/conditional-human-motion-generation'
SSH = ['ssh', '-b', '10.147.17.250', '-i', 'C:/Users/Pende/.ssh/id_ed25519',
       '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=10',
       '-o', 'ServerAliveCountMax=2', 'ouyanghao@10.147.17.32']

def execute(source, timeout=75):
    # Binary stdin avoids Windows command-line length and CRLF shell parsing.
    command = '/usr/bin/python3 -c '+shlex.quote('import sys;exec(sys.stdin.buffer.read())')
    result = subprocess.run(SSH+[command], input=source.encode('utf-8'),
                            capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).decode(errors='replace') or str(result.returncode))
    return result.stdout.decode('utf-8')

def sync(relative_paths):
    payload = {name: base64.b64encode((ROOT/name).read_bytes()).decode('ascii') for name in relative_paths}
    program = 'from pathlib import Path\nimport base64,hashlib,json\nr=Path('+repr(REMOTE)+')\ndata='+repr(payload)+'\n'
    program += '''out={}
for name,b64 in data.items():
 p=r/name
 p.resolve().relative_to(r.resolve())
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_bytes(base64.b64decode(b64))
 out[name]=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps(out))
'''
    hashes=json.loads(execute(program))
    import hashlib
    assert all(hashes[name]==hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in relative_paths)
    return hashes

"""Copy and publish one new recovery checkpoint pair; never delete server files.

Run once after starting the documented v4 continuation. The exclusive output
directory prevents duplicate workers. Download/remote and GitHub digests must
match before a receipt can report publication complete. Restricted raw data are
not touched. GitHub CLI uses the user's existing authentication.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--expected-updates',type=int,default=10061)
p.add_argument('--resume-failed',action='store_true')
a=p.parse_args()
if a.resume_failed:
    previous=json.loads((a.output/'receipt.json').read_text())
    if previous['status']!='FAILED' or previous['expected_updates']!=a.expected_updates:
        raise RuntimeError('Only an explicitly failed matching copy may be resumed')
    (a.output/f'failed_receipt_{time.time_ns()}.json').write_text(json.dumps(previous,indent=2)+'\n')
else:
    a.output.mkdir(parents=True,exist_ok=False)
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30',
     '-i','C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32']
run='/data/code/ouyanghao/oyh/paper-faithful-reproduction-20261022/training_fp64_sdpa_v4'
gh='C:/Program Files/GitHub CLI/gh.exe'
repo='Pendragon08/conditional-human-motion-generation'
tag='paper-reproduction-progress-20261023'
state={'status':'WAITING_FOR_NEW_CHECKPOINT','deletes_files':False,'expected_updates':a.expected_updates}

def write():
    (a.output/'receipt.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(state),flush=True)

def get_manifest():
    code=f'''from pathlib import Path
import json
p=Path({run!r})
rows=[]
for f in p.glob('checkpoint_ready_*.json'):
 r=json.loads(f.read_text())
 if r['optimizer_updates']=={a.expected_updates}:rows.append(r)
print(json.dumps(rows))
'''
    x=subprocess.run(ssh+['python3','-'],input=code,capture_output=True,text=True,encoding='utf-8',timeout=60)
    if x.returncode:raise RuntimeError(x.stderr)
    rows=json.loads(x.stdout)
    if len(rows)>1:raise RuntimeError('Ambiguous checkpoint manifests')
    return rows[0] if rows else None

try:
    write();deadline=time.monotonic()+7200
    consecutive_network_errors=0
    while time.monotonic()<deadline:
        try:
            manifest=get_manifest()
            consecutive_network_errors=0
            if state['status']=='WAITING_FOR_SSH_RETRY':
                state['status']='WAITING_FOR_NEW_CHECKPOINT';write()
        except (RuntimeError,subprocess.TimeoutExpired) as exc:
            consecutive_network_errors+=1
            state.update(status='WAITING_FOR_SSH_RETRY',last_network_error=str(exc),
                         consecutive_network_errors=consecutive_network_errors)
            write()
            if consecutive_network_errors>=10:raise
            time.sleep(30)
            continue
        if manifest:break
        time.sleep(30)
    else:raise TimeoutError('No matching ready checkpoint; no server files removed')
    if len(manifest['files'])!=2:raise ValueError('Expected exact model/optimizer pair')
    state.update(status='COPYING',manifest=manifest);write()
    copied=[]
    for row in manifest['files']:
        remote_path=PurePosixPath(row['file'])
        if remote_path.parent!=PurePosixPath(run)/'checkpoints':raise ValueError('Unexpected checkpoint parent')
        if not remote_path.name.startswith(('model','opt')) or remote_path.suffix!='.pt':raise ValueError('Unexpected file')
        name='reactive_v4_'+remote_path.name
        target=a.output/name
        if not target.exists():
            for attempt in range(1,6):
                partial=a.output/(name+f'.partial_{time.time_ns()}')
                try:
                    with partial.open('xb') as stream:
                        transfer=subprocess.run(ssh+['cat -- '+shlex.quote(str(remote_path))],stdout=stream,stderr=subprocess.PIPE,timeout=600)
                    if transfer.returncode:raise RuntimeError(transfer.stderr.decode(errors='replace'))
                    break
                except (RuntimeError,subprocess.TimeoutExpired) as exc:
                    state.setdefault('transfer_failures',[]).append({'file':name,'attempt':attempt,
                        'partial_preserved':str(partial),'bytes':partial.stat().st_size,'error':str(exc)})
                    write()
                    if attempt==5:raise
                    time.sleep(15)
            with partial.open('rb') as stream:
                digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest!=row['sha256'] or partial.stat().st_size!=row['bytes']:raise ValueError('Transferred checkpoint checksum mismatch')
            partial.rename(target)
        with target.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=row['sha256'] or target.stat().st_size!=row['bytes']:raise ValueError('Existing checkpoint checksum mismatch')
        copied.append({'name':name,'bytes':row['bytes'],'sha256':digest,'local_path':str(target)})
    if get_manifest()!=manifest:raise RuntimeError('Ready manifest changed during transfer')
    state.update(status='COPIED_AND_VERIFIED',copied=copied);write()
    for row in copied:
        existing=json.loads(subprocess.check_output([gh,'api',f'repos/{repo}/releases/tags/{tag}'],text=True,encoding='utf-8'))
        match=[x for x in existing['assets'] if x['name']==row['name']]
        if match:
            if len(match)!=1 or match[0]['size']!=row['bytes'] or match[0].get('digest')!='sha256:'+row['sha256']:
                raise RuntimeError('Existing GitHub asset differs; refusing overwrite')
        else:
            subprocess.run([gh,'release','upload',tag,row['local_path'],'--repo',repo],check=True,timeout=3600)
    release=json.loads(subprocess.check_output([gh,'api',f'repos/{repo}/releases/tags/{tag}'],text=True,encoding='utf-8'))
    assets={x['name']:x for x in release['assets']}
    for row in copied:
        asset=assets[row['name']]
        if asset['size']!=row['bytes'] or asset.get('digest')!='sha256:'+row['sha256']:
            raise RuntimeError('GitHub asset digest/size mismatch')
        row['url']=asset['browser_download_url']
    state.update(status='GITHUB_AND_LOCAL_VERIFIED',server_checkpoints_preserved=True);write()
except Exception as exc:
    state.update(status='FAILED',error=repr(exc));write();raise

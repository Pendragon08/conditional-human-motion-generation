"""Exact historical checkpoint cleanup under the user's 2026-10-24 request.

Only the eight listed files can be removed; latest 13001, datasets, source,
logs and original reproduction records are outside the whitelist.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import stat

DATA=Path('/data/code/ouyanghao/oyh')
HOME=Path('/home/ouyanghao')
ALLOWED={HOME/'reactivebfm-full-20261021/training_v1/checkpoints'/n
         for n in ['model000010000.pt','opt000010000.pt']}
ALLOWED|={DATA/'paper-faithful-reproduction-20261022/training_fp64_sdpa_v4/checkpoints'/n
          for n in ['model000010060.pt','opt000010060.pt','model000011000.pt',
                    'opt000011000.pt','model000012000.pt','opt000012000.pt']}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def check(row):
    p=Path(row['path']);assert p in ALLOWED and p.resolve()==p
    root=DATA if DATA in p.parents else HOME
    assert root.resolve()==root and os.getuid()==1013
    for q in [p,*p.parents]:assert not q.is_symlink(),str(q)
    s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==1013 and s.st_nlink==1
    assert (s.st_ino,s.st_mtime_ns,s.st_size)==(row['inode'],row['mtime_ns'],row['bytes'])
    assert sha(p)==row['sha256']
    a=row['github_asset'];assert a['url'].startswith('https://github.com/Pendragon08/conditional-human-motion-generation/releases/download/')
    assert a['bytes']==s.st_size and a['digest']=='sha256:'+row['sha256']
    assert row['local_restore_verified'] and row['local_restore_sha256']==row['sha256']
    return p,s

def active_check(paths):
    names={str(p) for p in paths};exceptions=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            comm=(proc/'comm').read_text().strip()
            args=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
            assert not any(n in args for n in names),'Candidate in active command'
            for fd in (proc/'fd').iterdir():
                try:v=os.readlink(fd)
                except FileNotFoundError:continue
                assert v not in names,'Candidate open by process'
            maps=(proc/'maps').read_text()
            assert not any(n in maps for n in names),'Candidate mapped by process'
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:
            assert comm in ['(sd-pam)','ssh-agent','gpg-agent','gnome-keyring-d','sshd'],comm
            exceptions.append({'pid':int(proc.name),'comm':comm})
    return exceptions

def main():
    a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--execute',action='store_true');args=a.parse_args()
    assert not args.output.exists()
    plan=json.loads(args.plan.read_text());rows=plan['files']
    assert {Path(r['path']) for r in rows}==ALLOWED and len(rows)==8
    paths=[];allocated={'data':0,'system':0}
    for row in rows:
        p,s=check(row);paths.append(p);allocated['data' if DATA in p.parents else 'system']+=s.st_blocks*512
    helpers=active_check(paths)
    import shutil
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status':'DRY_RUN_PASSED','mode':'EXECUTE' if args.execute else 'DRY_RUN',
        'files_verified':len(rows),'allocated_bytes':allocated,'removed':[],
        'protected_latest_optimizer_updates':13001,'datasets_selected':False,
        'logs_or_source_deleted':False,'known_os_helpers':helpers,
        'disk_before':{k:shutil.disk_usage(str(p))._asdict() for k,p in [('data',DATA),('system',HOME)]}}
    def save():args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    if args.execute:
        active_check(paths)
        for row in rows:
            p,s=check(row);p.unlink()
            receipt['removed'].append({'path':str(p),'bytes':row['bytes'],'sha256':row['sha256'],'github_asset':row['github_asset']});save()
        receipt['status']='VERIFIED_HISTORICAL_CHECKPOINTS_REMOVED'
    receipt['disk_after']={k:shutil.disk_usage(str(p))._asdict() for k,p in [('data',DATA),('system',HOME)]}
    save();print(json.dumps(receipt))

if __name__=='__main__':main()

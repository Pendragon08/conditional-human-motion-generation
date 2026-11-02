"""Restore one privately preserved pip wheel-cache body; no installed package edits.

Predelete evidence: SciPy1.16.2 cached wheel, 38550641 bytes, ZIP CRC and both
source/D SHA verified, native ownership/no-reparse checked. This is recovery
code for a disposable download cache, not experiment data or a checkpoint.
The private body remains on D and is never uploaded by this tool.
"""
import argparse,base64,hashlib,json,os,pathlib,shutil,subprocess

SHA='0ce54e07bbb394b417457409a64fd015be623f36e330ac49306433ffe04bc97e'
SIZE=38550641
SUFFIX='AppData/Local/pip/Cache/http-v2/1/e/b/3/2/1eb3215580710c32fd3833e34bd53e931a4f3e8c1215a52ab78394a1.body'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--backup',required=True);p.add_argument('--restore',action='store_true');a=p.parse_args()
    src=pathlib.Path(a.backup);dst=pathlib.Path.home()/SUFFIX
    assert os.name=='nt' and src.is_absolute() and src.resolve()==src and src.is_file()
    assert src.stat().st_size==SIZE and digest(src)==SHA
    if not a.restore:
        print(json.dumps(dict(status='PRIVATE_SOFTWARE_CACHE_BACKUP_SHA_VERIFIED',bytes=SIZE,sha256=SHA,source_modified=False)));return
    assert dst.resolve()==dst and dst.is_relative_to(pathlib.Path.home())
    assert dst.parent.is_dir() and shutil.disk_usage(dst.parent).free>=SIZE+2*1024**3
    paths=[src,dst.parent]
    for item in [src,dst.parent]:
        here=item
        while here!=here.parent:
            assert not here.is_symlink() and here.resolve()==here
            here=here.parent
    encoded=base64.b64encode(json.dumps([str(x) for x in paths]).encode()).decode()
    script="[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false);$ErrorActionPreference='Stop';$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value;$paths=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('"+encoded+"'))|ConvertFrom-Json;$rows=@($paths|ForEach-Object{$i=Get-Item -LiteralPath $_ -Force;$acl=Get-Acl -LiteralPath $_;[pscustomobject]@{owned=($acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -eq $sid);linked=(($i.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)}});ConvertTo-Json -Compress -InputObject $rows"
    z=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',base64.b64encode(script.encode('utf-16le')).decode()],capture_output=True,text=True,encoding='utf-8',check=True,timeout=30)
    assert all(x['owned'] and not x['linked'] for x in json.loads(z.stdout))
    if dst.exists():
        assert dst.stat().st_size==SIZE and digest(dst)==SHA
    else:
        with src.open('rb') as f,dst.open('xb') as out:
            shutil.copyfileobj(f,out,4*1024**2);out.flush();os.fsync(out.fileno())
        assert dst.stat().st_size==SIZE and digest(dst)==SHA
    assert digest(src)==SHA
    print(json.dumps(dict(status='PIP_DOWNLOAD_CACHE_RESTORED_EXACTLY',bytes=SIZE,sha256=SHA,backup_preserved=True,installed_packages_modified=False)))
if __name__=='__main__':main()

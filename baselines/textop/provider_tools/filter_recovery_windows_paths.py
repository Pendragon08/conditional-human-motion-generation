"""Read native Windows ownership; reject linked/foreign destination ancestry."""
import base64
import json
import os
import pathlib
import subprocess


def powershell_json(script, timeout=30):
    encoded=base64.b64encode(script.encode('utf-16le')).decode('ascii')
    native=pathlib.Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0'
    env=os.environ.copy();env['PSModulePath']=str(native/'Modules')
    p=subprocess.run([str(native/'powershell.exe'),'-NoProfile','-NonInteractive','-EncodedCommand',encoded],
                     capture_output=True,text=True,encoding='utf-8',timeout=timeout,env=env)
    assert p.returncode==0,p.stderr
    return json.loads(p.stdout)


def guard_local(root, target, directory=False, must_absent=False):
    root,target=pathlib.Path(root),pathlib.Path(target)
    assert os.name=='nt' and root.is_absolute() and root.is_dir() and root.resolve()==root
    assert target.is_absolute() and target.resolve()==target and target.drive==root.drive
    relative=target.relative_to(root);assert '..' not in relative.parts
    if must_absent:assert not os.path.lexists(str(target)),'Preserve existing partial/final file'
    paths=[root];current=root
    for index,part in enumerate(relative.parts):
        current=current/part
        if os.path.lexists(str(current)):
            assert not current.is_symlink() and current.resolve()==current
            if index<len(relative.parts)-1 or directory:assert current.is_dir()
            else:assert current.is_file()
            paths.append(current)
    payload=base64.b64encode(json.dumps([str(x) for x in paths],ensure_ascii=False).encode('utf-8')).decode('ascii')
    script="""[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$ErrorActionPreference='Stop'
$paths = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('%s')) | ConvertFrom-Json
$sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$rows = @($paths | ForEach-Object {
 $acl = Get-Acl -LiteralPath $_
 $item = Get-Item -LiteralPath $_ -Force
 [pscustomobject]@{owner_matches=($acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -eq $sid);reparse=(($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)}
})
[pscustomobject]@{entries=$rows;count=$rows.Count} | ConvertTo-Json -Depth 5 -Compress
""" % payload
    result=powershell_json(script)
    assert result['count']==len(paths) and all(r['owner_matches'] and not r['reparse'] for r in result['entries']), 'Backup ancestry must belong to current Windows user without reparse points'
    return dict(current_user_owned=True,reparse_points=False,existing_components_checked=len(paths),path=str(target))

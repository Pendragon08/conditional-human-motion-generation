"""Windows-owned synthetic paths only. No SSH, model, copy or data deletion."""
import argparse,base64,json,os,pathlib
import filter_recovery_windows_paths as mod
from unittest.mock import patch
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
assert a.run.isascii() and a.run.isidentifier()
D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
out=D/'diagnostics/filter_recovery_checkpoint_paths_20261027'/a.run
assert not out.exists();out.mkdir(parents=True)
checks=[]
def reject(name,fn):
 try:fn()
 except (AssertionError,ValueError):checks.append(name);return
 raise AssertionError('Unexpected acceptance: '+name)
mod.guard_local(D,out,directory=True);checks.append('real_current_user_owned_path')
target=out/'future'/'checkpoint.pt';mod.guard_local(D,target,must_absent=True);checks.append('new_owned_destination_no_write')
original=out/'original.txt';original.write_text('own synthetic fixture',encoding='utf-8')
mod.guard_local(D,original);checks.append('existing_owned_file')
partial=out/'checkpoint.partial';partial.write_text('retained partial',encoding='utf-8')
reject('existing_partial_rejected',lambda:mod.guard_local(D,partial,must_absent=True))
reject('outside_root_rejected',lambda:mod.guard_local(out,out.parent/'escape'))
reject('regular_file_ancestor_rejected',lambda:mod.guard_local(D,original/'child'))
normal=out/'normal';normal.mkdir()
payload=base64.b64encode(json.dumps({'path':str(out/'junction'),'target':str(normal)},ensure_ascii=False).encode('utf-8')).decode()
script="""[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false)
$ErrorActionPreference='Stop'
$j=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('%s'))|ConvertFrom-Json
$null=New-Item -ItemType Junction -Path $j.path -Target $j.target
@{created=$true}|ConvertTo-Json -Compress
"""%payload
assert mod.powershell_json(script)['created']
reject('real_ntfs_junction_rejected',lambda:mod.guard_local(D,out/'junction'/'child'))
with patch.object(mod,'powershell_json',return_value={'count':1,'entries':[{'owner_matches':False,'reparse':False}]}):
 reject('synthetic_foreign_owner_rejected',lambda:mod.guard_local(D,D,directory=True))
with patch.object(mod,'powershell_json',return_value={'count':1,'entries':[{'owner_matches':True,'reparse':True}]}):
 reject('synthetic_reparse_attribute_rejected',lambda:mod.guard_local(D,D,directory=True))
result=dict(status='RECOVERY_WINDOWS_PATH_FIXTURES_PASSED',checks=len(checks),names=checks,
 real_owner_and_junction_checked=True,foreign_owner_and_reparse_metadata_simulated=True,
 fixtures_retained=str(out),data_deleted=False,ssh_calls=0,sdk_runs=0,checkpoint_loads=0)
(out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
B=pathlib.Path(__file__).resolve().parents[1];public=B/'results/filter_recovery_checkpoint_tools_20261027'/(a.run+'.json')
assert not public.exists();public.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))

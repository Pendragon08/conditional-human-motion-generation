import sys,pathlib,json
TASK=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parent/'paper_faithful_reproduction_20261022'))
from session import remote
print(remote(r'''
import pathlib,json,shutil,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
for p in sorted((r/'logs').glob('*.log')):
    with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4500));s=f.read().decode(errors='replace')
    print('\nLOG',p.name,s[-4500:])
print('RECEIPTS',[p.name for p in (r/'receipts').glob('*')])
print('DISKS',shutil.disk_usage(r),shutil.disk_usage('/home/ouyanghao'))
partials=list(r.rglob('*.partial'))+list(pathlib.Path('/home/ouyanghao/textop-assets-20261024').rglob('*.partial'))
print('PARTIAL',[(str(p.relative_to(r) if str(p).startswith(str(r)) else p),p.stat().st_size) for p in partials])
''',timeout=40))

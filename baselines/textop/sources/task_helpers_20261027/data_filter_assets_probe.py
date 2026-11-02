import pathlib,json,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
import os
found=[]
for root,dirs,files in os.walk('/data/code/ouyanghao/oyh'):
    dirs[:]=[x for x in dirs if x not in ['.git','site-packages','node_modules','__pycache__']]
    for name in files:
        if 'babel' in name.lower():found.append(str(pathlib.Path(root)/name))
print('BABEL_NAMED_FILES',json.dumps(found)[:10000])
print('RECEIPTS',json.dumps([x.name for x in (r/'receipts').glob('*')]))
for x in (r/'assets').rglob('env.yaml'):
    print('TRACKER_ENV',str(x));s=x.read_text()
    for key in ['Priv','actor','critic','domain','anchor_pos','ee_body_pos']:
        lines=s.splitlines();hits=[i for i,line in enumerate(lines) if key in line]
        for i in hits[:4]: print('\n'.join(lines[max(0,i-1):i+5]))
print('GMR_STATUS',(r/'data_preparation_20261025/available500_v1/progress.json').read_text())

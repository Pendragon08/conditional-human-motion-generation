import pathlib,json
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
for p in [r/'data_preparation_20261025/audit_summary.json',r/'logs/data_audit.log']:
    if p.exists():print(p.name,p.read_text()[-14000:])
for p in (r/'data_preparation_20261025').glob('*'):print('OUTPUT',p.name,p.stat().st_size)

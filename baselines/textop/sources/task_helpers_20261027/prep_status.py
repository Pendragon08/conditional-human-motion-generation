import pathlib,json
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
for name in ['smoke_v2','available500_v1']:
    p=r/'data_preparation_20261025'/name
    for f in ['progress.json','failures.private.jsonl']:
        q=p/f
        if q.exists():print(name,f,q.read_text()[-6000:])
for name in ['data_prep_smoke_v2.log','data_prep500.log']:
    p=r/'logs'/name
    if p.exists():print(name,p.read_text()[-2000:])

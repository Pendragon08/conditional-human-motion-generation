import pathlib,subprocess,json,os
root=pathlib.Path('/data/code/ouyanghao/oyh');targets=[root,pathlib.Path('/home/ouyanghao/.cache')]
out={'scope':'read-only inventory; no files deleted','largest_owned_subfolders':[]}
for target in targets:
    if not target.exists():continue
    p=subprocess.run(['du','-x','-B1','--max-depth=2',str(target)],capture_output=True,text=True,timeout=110)
    rows=[]
    for line in p.stdout.splitlines():
        size,path=line.split('\t',1);q=pathlib.Path(path)
        if q.stat().st_uid==os.getuid():rows.append({'bytes':int(size),'path':path})
    out['largest_owned_subfolders']+=sorted(rows,key=lambda x:x['bytes'],reverse=True)[:24]
(root/'textop-reproduction-20261024/receipts/storage_data_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))

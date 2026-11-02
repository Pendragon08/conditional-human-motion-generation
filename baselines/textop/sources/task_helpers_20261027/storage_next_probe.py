import pathlib,subprocess,json
root=pathlib.Path('/data/code/ouyanghao/oyh')
result=subprocess.run(['du','-x','-B1','--max-depth=2',str(root)],capture_output=True,text=True)
rows=[]
for line in result.stdout.splitlines():
    value,path=line.split('\t',1);rows.append({'bytes':int(value),'path':path})
rows.sort(key=lambda x:x['bytes'],reverse=True)
print(json.dumps({'largest':rows[:32]}))
old=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024/receipts/storage_data_audit.json')
if old.exists():print('PREVIOUS_AUDIT',old.read_text()[-7000:])

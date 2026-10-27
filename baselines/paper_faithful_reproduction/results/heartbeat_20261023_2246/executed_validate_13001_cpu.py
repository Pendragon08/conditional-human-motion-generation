from session import *
import torch
torch.set_num_threads(4)
root=LOCAL/'heartbeat_20261023_2246'
backup=json.loads((root/'checkpoint_13001_backup_receipt.json').read_text(encoding='utf-8'))
assert backup['status']=='GITHUB_AND_LOCAL_VERIFIED'
rows=[]
for row in backup['copied']:
    p=Path(row['local_path'])
    with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256']
    obj=torch.load(p,map_location='cpu',weights_only=False)
    tensors=[];steps=[]
    def walk(x,path=''):
        if torch.is_tensor(x):
            assert bool(torch.isfinite(x).all()),path
            tensors.append(path)
            if path.endswith('/step'):steps.append(int(x.item()))
        elif isinstance(x,dict):
            for k,v in x.items():walk(v,path+'/'+str(k))
        elif isinstance(x,(list,tuple)):
            for i,v in enumerate(x):walk(v,path+'/'+str(i))
    walk(obj)
    counts={key:sum(path.startswith('/'+str(key)+'/') for path in tensors) for key in obj}
    item={'name':p.name,'tensor_count':len(tensors),'finite_tensor_counts':counts,'finite':True,'top_level_keys':list(obj) if isinstance(obj,dict) else None,'optimizer_steps_count':len(steps),'optimizer_steps_unique':sorted(set(steps))}
    if 'opt' in p.name:assert len(steps)==365 and set(steps)=={13001} and len(tensors)==1095
    else:assert counts=={'model':388,'model_avg':388}
    rows.append(item)
    del obj
result={'utc':utc(),'status':'CPU_DESERIALIZED_ALL_TENSORS_FINITE','torch_version':torch.__version__,'files':rows,'cpu_validation_not_server_numerical_equivalence':True}
dump(root/'checkpoint_finite_validation.json',result)
print(json.dumps(result,ensure_ascii=False))

import sys,pathlib,json
TASK=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parent/'paper_faithful_reproduction_20261022'))
from session import remote,REPO
script=r'''
import pathlib,torch,json,hashlib
from robotmdar.model.mld_vae import AutoMldVae
torch.set_num_threads(1)
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=next((r/'runs').rglob('ckpt_1000.pth'))
c=torch.load(p,map_location='cpu');assert c['step']==1000
model=AutoMldVae(nfeats=57,latent_dim=[1,128],h_dim=512,ff_size=1024,num_layers=9,num_heads=4)
model.load_state_dict(c['vae'],strict=True)
opt=torch.optim.Adam(model.parameters(),lr=1e-4);opt.load_state_dict(c['optimizer'])
vf=all(torch.isfinite(t).all().item() for t in c['vae'].values())
of=all(torch.isfinite(t).all().item() for st in c['optimizer']['state'].values() for t in st.values() if torch.is_tensor(t))
steps=[int(st['step']) for st in c['optimizer']['state'].values()]
assert vf and of
out={'checkpoint':str(p),'loop_step':c['step'],'optimizer_param_step_min':min(steps),'optimizer_param_step_max':max(steps),'model_all_finite':vf,'optimizer_all_finite':of,'strict_model_load':True,'optimizer_load':True,'validation':'CPU state load only; no new optimizer update, no quality claim'}
(r/'receipts/checkpoint_load_witness.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
'''
out=json.loads(remote(script,python='/data/code/ouyanghao/oyh/textop-reproduction-20261024/env/bin/python',timeout=120))
(REPO/'baselines/textop/results/checkpoint_load_witness.json').write_text(json.dumps(out,indent=2))
pathlib.Path('D:/具身智能/TextOp复现_20261024/checkpoint_load_witness.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))

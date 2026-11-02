import pathlib,json,subprocess,os
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
code=r'''
import pathlib,json,joblib,numpy as np
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
d=r/'upstream/TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps'
out={}
for split in ['train','val']:
    data=joblib.load(d/(split+'.pkl'));x=data[0]
    out[split]={'count':len(data),'keys':list(x),'sample':{k:repr(v)[:500] for k,v in x.items() if k!='motion'},'motion':{k:{'type':type(v).__name__,'shape':list(v.shape) if hasattr(v,'shape') else None,'repr':repr(v)[:120] if not hasattr(v,'shape') else None} for k,v in x['motion'].items()}}
    out[split]['last_names']=[x.get('feat_p') for x in data[-8:]]
    del data
out['statistics']= (d/'statistics.yaml').read_text()
out['source_environment']={}
root=pathlib.Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017')
for p in [root/'envs/gmr_amass_20261021',root/'data/amass/ssm_20261021',root/'data/amass/remaining_20261021_v4',root/'data/amass/remaining_20261021_v5']:
    out['source_environment'][str(p)]=[str(x) for x in list(p.iterdir())[:10]] if p.exists() else None
print(json.dumps(out))
'''
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
p=subprocess.run([str(r/'env/bin/python'),'-c',code],capture_output=True,text=True,env=env,timeout=120)
assert p.returncode==0,p.stderr
(r/'receipts/data_schema_probe.json').write_text(p.stdout)
print(p.stdout)

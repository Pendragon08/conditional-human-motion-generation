import pathlib,json,subprocess,os
root=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/data')
out={'top':[x.name for x in root.iterdir()],'raw_top':[x.name for x in list((root/'raw').iterdir())[:15]],'metadata_files':[]}
for p in root.glob('*.json'):out['metadata_files'].append(str(p))
for p in (root/'raw').glob('*.json'):out['metadata_files'].append(str(p))
py='/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python'
code=r'''
import pathlib,numpy as np,json
r=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/data/raw');files=list(r.glob('*.npz'));out={'count':len(files),'examples':[]}
for p in files[:3]:
    with np.load(p,allow_pickle=False) as x:out['examples'].append({'path':str(p),'fps':float(x['fps']),'qpos_shape':list(x['qpos'].shape)})
print(json.dumps(out))
'''
p=subprocess.run([py,'-c',code],capture_output=True,text=True);out['robot_npz']=p.stdout;out['error']=p.stderr[-500:]
p=subprocess.run(['du','-x','-B1','--max-depth=2','/data/code/ouyanghao/oyh/humanoid-combat-motion/cache'],capture_output=True,text=True);rows=[(int(x.split('\t',1)[0]),x.split('\t',1)[1]) for x in p.stdout.splitlines()];out['cache_largest']=sorted(rows,reverse=True)[:10]
print(json.dumps(out))

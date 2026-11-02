import pathlib,subprocess,json,collections,os
r=pathlib.Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/cache')
out={}
for p in r.iterdir():
 if not p.is_dir() or p.is_symlink():continue
 x=subprocess.run(['du','-x','-B1','--max-depth=3',str(p)],capture_output=True,text=True,timeout=30)
 out[p.name]=sorted([{'bytes':int(z.split('\t')[0]),'path':z.split('\t')[1]} for z in x.stdout.splitlines()],key=lambda a:a['bytes'],reverse=True)[:12]
for p in [pathlib.Path('/home/ouyanghao/.cache/pip'),pathlib.Path('/home/ouyanghao/.cache/ov'),pathlib.Path('/home/ouyanghao/.nv/ComputeCache')]:
 if p.exists():
  x=subprocess.run(['du','-x','-B1','--max-depth=2',str(p)],capture_output=True,text=True,timeout=30);out[str(p)]=x.stdout.splitlines()[:12]
print(json.dumps(out))

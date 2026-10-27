import pathlib,os,json,collections,zipfile
root=pathlib.Path('/data/code/ouyanghao/oyh')
out={}
for rel in ['conditional-human-motion-generation/artifacts','conditional-human-motion-generation/results','humanoid-combat-motion/data','humanoid-combat-motion/cache','humanoid-combat-motion/third_party']:
    base=root/rel;stats=collections.Counter();examples=[];top=[x.name for x in base.iterdir()] if base.exists() else []
    for path,dirs,files in os.walk(base):
        dirs[:]=[d for d in dirs if d not in ['.git','site-packages','node_modules','__pycache__']]
        for f in files:
            p=pathlib.Path(path)/f
            if p.suffix not in ['.npz','.pkl','.npy','.bz2','.zip','.gz']:continue
            stats[p.suffix]+=1
            if p.suffix=='.npz' and len(examples)<8:
                try:
                    with zipfile.ZipFile(p) as z:keys=z.namelist()[:24]
                    examples.append({'file':str(p),'keys':keys})
                except Exception:pass
    out[rel]={'top':top[:30],'suffix_counts':dict(stats),'npz_examples':examples}
print(json.dumps(out))

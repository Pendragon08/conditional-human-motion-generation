import pathlib,json,os,signal,subprocess,shutil,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
out={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
launch=json.loads((r/'receipts/training_launch.json').read_text());pid=launch['pid'];proc=pathlib.Path(f'/proc/{pid}')
if proc.exists() and str(r/'scripts/train_official.py') in (proc/'cmdline').read_bytes().decode(errors='replace'):
    assert proc.stat().st_uid==1013
    os.kill(pid,signal.SIGSTOP)
    out['textop_training']={'pid':pid,'action':'SIGSTOP preserves live training state; prioritize dataset preparation','state':next(x for x in (proc/'status').read_text().splitlines() if x.startswith('State:'))}
else:out['textop_training']={'pid':pid,'action':'not running; no signal sent'}
for name in ['training_progress.json','training_exit.json','download_complete.json']:
    p=r/'receipts'/name
    if p.exists():
        obj=json.loads(p.read_text());out[name]={'files':len(obj.get('files',[])),'bytes':obj.get('total_bytes')} if name=='download_complete.json' else obj
out['disk']={str(p):shutil.disk_usage(p)._asdict() for p in [r,pathlib.Path('/home/ouyanghao')]}
root=pathlib.Path('/data/code/ouyanghao/oyh');hits=[];raw_groups={}
for base,dirs,files in os.walk(root):
    dirs[:]=[d for d in dirs if d not in ['site-packages','.git','node_modules','__pycache__'] and not d.startswith('.venv') and d not in ['env','venv']]
    b=pathlib.Path(base)
    for f in files:
        low=f.lower();p=b/f
        if low in ['smplx_neutral.npz','smplx_male.npz','smplx_female.npz'] or (low.endswith('.json') and ('babel' in str(p).lower()) and low in ['train.json','val.json']):hits.append(str(p))
        if low.endswith('.npz') and any(k in str(b).lower() for k in ['amass','raw_assets','raw_motions','原始资产']):
            key=str(b.relative_to(root)).split('/')[:4];key='/'.join(key)
            raw_groups[key]=raw_groups.get(key,0)+1
    for d in dirs:
        if any(k in d.lower() for k in ['gmr','babel','smplx','amass']):hits.append(str(b/d))
out['asset_candidates']=hits[:160];out['raw_npz_groups']=raw_groups
(r/'receipts/data_preparation_pause.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))

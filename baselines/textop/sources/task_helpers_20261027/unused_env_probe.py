import pathlib,subprocess,json,os,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh');out={}
for rel in ['humanoid-combat-motion/envs','conditional-human-motion-generation/cache','humanoid-combat-motion/.cache','humanoid-combat-motion/.venv-engineai-official']:
 p=r/rel
 if p.exists():
  z=subprocess.run(['du','-x','-B1','--max-depth=2',str(p)],capture_output=True,text=True,timeout=50)
  out[rel]=sorted([{'bytes':int(x.split('\t')[0]),'path':x.split('\t')[1]} for x in z.stdout.splitlines()],key=lambda a:a['bytes'],reverse=True)[:8]
out['owned_processes']=[]
for d in pathlib.Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  if d.stat().st_uid!=os.getuid():continue
  cmd=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if cmd:out['owned_processes'].append({'pid':int(d.name),'command':cmd[:400]})
 except OSError:pass
print(json.dumps(out))

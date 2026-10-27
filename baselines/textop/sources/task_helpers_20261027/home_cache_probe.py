import pathlib,subprocess,json,os
r=pathlib.Path('/home/ouyanghao');assert r.stat().st_uid==1013
p=subprocess.run(['du','-x','-B1','--max-depth=2',str(r)],capture_output=True,text=True,timeout=80)
rows=[]
for line in p.stdout.splitlines():
 n,s=line.split('\t',1)
 try:
  if pathlib.Path(s).stat().st_uid==1013:rows.append({'path':s,'bytes':int(n)})
 except OSError:pass
print(json.dumps({'largest_home':sorted(rows,key=lambda x:x['bytes'],reverse=True)[:20]}))

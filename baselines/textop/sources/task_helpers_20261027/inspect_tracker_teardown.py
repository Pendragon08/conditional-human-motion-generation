import pathlib,json,os
r=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/isaacsim-4.5.0/exts')
for folder in r.glob('*simulation_app*'):
 for p in folder.rglob('simulation_app.py'):
  s=p.read_text();at=s.find('    def close(')
  if at>=0:print('CLOSE_SOURCE',str(p),s[at:at+8500])
rows=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=1013:continue
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'tracker_kernel_witness.py' in cmd:
   rows.append({'pid':int(p.name),'command':cmd,'state':next(x for x in (p/'status').read_text().splitlines() if x.startswith('State:'))})
 except OSError:pass
print('DIAGNOSTIC_PROCESSES',json.dumps(rows))

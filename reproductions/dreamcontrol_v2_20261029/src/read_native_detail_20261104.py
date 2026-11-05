import pathlib,json
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');N=R/'stage3_supermimic29_20261103/native_universal_20261104'
for s in ['continuation_status.json','capacity_8912_v1_guard.json','capacity_8912_v1/status.json','benchmark_8912_v1_guard.json','benchmark_8912_v1/status.json']:
 p=N/s
 if p.exists(): print(s,p.read_text()[-5000:])
for s in ['capacity_8912_v1.log','benchmark_8912_v1.log']:
 p=N/s
 if p.exists(): print(s,'\n'.join(p.read_text().splitlines()[-5:]))
for name,terms in [('train.py',['collect','sim_steps','resume','warmup','wm_batches','steps_per','checkpoint','ckpt','ctrl_freq']),('evaluate.py',[''])]:
 p=N/'source/supermimic'/name
 if p.exists():
  lines=p.read_text().splitlines()
  for i,line in enumerate(lines):
   if any(t in line for t in terms): print(name,i+1,line)
print('FILES',[p.name for p in (N/'source/supermimic').glob('*.py')])

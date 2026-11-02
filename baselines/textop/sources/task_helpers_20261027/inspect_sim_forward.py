import pathlib,json
p=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py')
s=p.read_text()
for term in ['    def forward(', '    def render(', '    def stop(']:
 a=s.find(term)
 if a>=0:print(term,s[a:a+2000])
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=r/'receipts/tracker_kernel_author_v2_exit.json';print('EXIT_V2',p.read_text() if p.exists() else 'PENDING')
log=r/'logs/tracker_kernel_author_v2.log'
with log.open('rb') as f:f.seek(max(0,log.stat().st_size-1800));print('TAIL',f.read().decode(errors='replace'))

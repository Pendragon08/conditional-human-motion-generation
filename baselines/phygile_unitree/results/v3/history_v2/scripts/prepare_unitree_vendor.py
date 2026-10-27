"""Create fresh, pinned official framework checkouts without editing them."""
from pathlib import Path
import argparse,json,os,subprocess
p=argparse.ArgumentParser();p.add_argument('--framework',choices=['lab','mjlab'],required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
repo,commit={'lab':('unitree_rl_lab','4960b84732b0c2ec593dccbfe963fda1bcd7b1e3'),'mjlab':('unitree_rl_mjlab','1425b15f73bd4095f0df53709d7c389c3eb9e790')}[a.framework]
if a.destination.exists():raise FileExistsError('Use a fresh destination; existing checkouts are preserved')
env=os.environ.copy();env['GIT_TERMINAL_PROMPT']='0'
subprocess.run(['git','clone','https://github.com/unitreerobotics/'+repo+'.git',str(a.destination)],env=env,check=True)
subprocess.run(['git','-C',str(a.destination),'checkout','--detach',commit],check=True)
head=subprocess.check_output(['git','-C',str(a.destination),'rev-parse','HEAD'],text=True).strip()
if head!=commit:raise RuntimeError('Framework checkout SHA mismatch')
print(json.dumps({'repository':repo,'commit':head,'destination':str(a.destination),'upstream_modified':False}))

"""Read-only progress, process and actual-render metadata; never starts a job."""
import json
from r3_world_ssh_transport import REMOTE, execute

program = r'''
from pathlib import Path
import json,time,subprocess,hashlib
r=Path(REMOTE)
out={'observed_unix':time.time(),'read_only':True}
prefix=r/'artifacts/monitoring/r3_world_ppo_formal_20261007'
for suffix in ['.state','.dispatch.json']:
 p=prefix.with_suffix(suffix)
 if p.exists():
  out[suffix]={'mtime':p.stat().st_mtime,'text':p.read_text()}
stage={}
if prefix.with_suffix('.state').exists():
 try:stage=json.loads(prefix.with_suffix('.state').read_text())
 except Exception as error:out['state_parse_error']=repr(error)
if stage.get('status')=='RUNNING':
 mode=stage['stage'];arm=stage['arm'];seed=stage['seed']
 p=r/f'artifacts/monitoring/r3_world_ppo_{mode}_{arm}_{seed}_20261007'
 for suffix in ['.state','.log']:
  f=p.with_suffix(suffix)
  if f.exists():
   lines=f.read_text(errors='replace').splitlines()
   updates=[line for line in lines if line.startswith(('WORLD_PPO_UPDATE','WORLD_PPO_EVAL_STEP'))]
   out['stage'+suffix]={'mtime':f.stat().st_mtime,'bytes':f.stat().st_size,
    'latest_update':updates[-1] if updates else None,'tail':lines[-12:]}
 directory=r/f'artifacts/world_residual_ppo_20261007/formal/{mode}_{arm}_seed{seed}_v1'
 p=directory/'progress.json'
 if p.exists():out['progress']=json.loads(p.read_text())
 out['checkpoints']=[{'name':p.name,'size':p.stat().st_size,'mtime':p.stat().st_mtime}
  for p in sorted(directory.glob('model_update*.pt'))]
for filename in ['r3_i1_live_viewer.state','r3_i1_live_frame.json','r3_active_dreamcontrol_training.json']:
 p=r/'artifacts/monitoring'/filename
 if p.exists():out[filename]={'mtime':p.stat().st_mtime,'text':p.read_text()}
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu',
 '--format=csv,noheader,nounits'],text=True)
out['tmux']=subprocess.run(['tmux','list-sessions'],capture_output=True,text=True).stdout
ps=subprocess.check_output(['ps','-eo','pid,ppid,etimes,pcpu,stat,args'],text=True)
out['processes']=[line for line in ps.splitlines() if any(s in line for s in
 ['r3_world_ppo_control_formal_v3_20261007.py','r3_world_ppo_eval_isaac_20261007.py',
  'r3_world_ppo_formal_queue_20261007.py','r3_i1_live_training_viewer.py']) and '/usr/bin/python3 -c' not in line]
for pid in [1589572,3854248]:
 p=Path('/proc')/str(pid)/'stat'
 if p.exists():
  fields=p.read_text().split()
  out[f'viewer_pid_{pid}']={'exists':True,'start_ticks':fields[21]}
 else:out[f'viewer_pid_{pid}']={'exists':False}
print(json.dumps(out,ensure_ascii=True))
'''.replace('REMOTE', repr(REMOTE))
data=json.loads(execute(program))
print(json.dumps(data,ensure_ascii=True,indent=2))

"""Read actual rolling training timings; no training/source changes."""
import json,statistics,datetime,os,shutil,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';D=R/'stage3_supermimic29_20261103'
PPO=S/'runs/pick500_8912_3000_v1';SM=D/'runs/pick500_8912_72000_parallel_v2';W=D/'parallel_pick500_20261104'
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines(keepends=True) if x.endswith('\n')]
def dist(v):
 v=sorted(v);return {'count':len(v),'mean':statistics.mean(v),'median':statistics.median(v),'p10':v[int((len(v)-1)*.1)],'p90':v[int((len(v)-1)*.9)]}
def alive(pid):
 p=Path('/proc',str(pid),'stat');return p.exists() and p.read_text().split()[2]!='Z'
now=datetime.datetime.now(datetime.timezone.utc);tz=datetime.timezone(datetime.timedelta(hours=8))
out={'as_of_Beijing':now.astimezone(tz).isoformat(),'runtime_disk_floor_bytes':8*1024**3+356*1024**2,'free_bytes':shutil.disk_usage('/data').free}
p=read(PPO/'status.json');pm=rows(PPO/'metrics.jsonl');s=read(SM/'status.json');sm=rows(SM/'metrics.jsonl')
out['PPO']={'status':p['status'],'pid':p['pid'],'alive':alive(p['pid']),'completed':p['policy_updates'],'target':3000,
 'metrics_age_seconds':now.timestamp()-(PPO/'metrics.jsonl').stat().st_mtime,'windows':{}}
for n in [20,50,100]:
 v=[x['collection_time_s']+x['learn_time_s'] for x in pm[-n:]];a=dist(v);a['remaining_hours']=(3000-p['policy_updates'])*a['mean']/3600;out['PPO']['windows'][str(n)]=a
cp=sorted((PPO/'ppo').glob('*.pt'),key=lambda x:x.stat().st_mtime)
out['PPO']['checkpoint_count']=len(cp)
if len(cp)>1:
 a,b=cp[-2:];delta=int(b.stem.split('_')[-1])-int(a.stem.split('_')[-1]);out['PPO']['last_checkpoint_wall_seconds_per_update']=(b.stat().st_mtime-a.stat().st_mtime)/delta
out['SuperMimic']={'status':s['status'],'pid':s['pid'],'alive':alive(s['pid']),'progress':s.get('progress'),'counters':s.get('counters'),
 'error':s.get('error'),'metrics_age_seconds':now.timestamp()-(SM/'metrics.jsonl').stat().st_mtime,'windows':{}}
for n in [10,20,50]:
 values=[]
 for x,y in zip(sm[-n-1:-1],sm[-n:]):
  steps=y['actual_vector_steps']-x['actual_vector_steps'];seconds=y['elapsed_seconds']-x['elapsed_seconds']
  if steps>0 and x['actual_vector_steps']>=96:values.append(seconds/steps)
 if values:
  a=dist(values);a['remaining_hours']=(72000-s['progress']['actual_vector_steps'])*a['mean']/3600;out['SuperMimic']['windows'][str(n)]=a
out['PPO_controller']={k:v for k,v in read(W/'controller_v1/status.json').items() if k in ['status','pid','ppo_status','ppo_updates','supermimic_status']}
out['SM_controller']={k:v for k,v in read(W/'numerical_repair_v2/status.json').items() if k in ['status','pid','last_phase_status','progress','counters','error']}
out['SM_eval_waiter']=read(W/'numerical_repair_v2/eval_followup_status.json')
out['PPO_evaluation_exists']=(R/'stage3_skill_sequence_20261103/evaluations/pick500_final_3000_v1/report.json').exists()
out['GPU']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],universal_newlines=True)
out['latest_checkpoint']=s.get('latest_checkpoint')
print(json.dumps(out,indent=2))

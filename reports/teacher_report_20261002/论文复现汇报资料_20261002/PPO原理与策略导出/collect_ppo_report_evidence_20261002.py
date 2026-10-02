"""Read existing logs and remote source/files; never instantiate or mutate training."""
import csv, datetime, hashlib, json, math, shlex, subprocess
from pathlib import Path
import numpy as np

REPO=Path(r'D:\具身智能\conditional-human-motion-generation')
ROOT=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
OUT=REPO/'results/ppo_training_interpretation_20261002'
OUT.mkdir(parents=True,exist_ok=True)
remote_root='/data/code/ouyanghao/oyh/conditional-human-motion-generation'
library=remote_root+'/third_party/isaacsim-4.5.0/kit/python/lib/python3.10/site-packages/rsl_rl'
runs={'Kick':('/data/code/ouyanghao/oyh/humanoid-combat-motion','2026-09-27_03-19-31-Kick-a'),
      'Punch':(remote_root,'2026-09-29_12-37-46-Punch-a'),
      'Jump':(remote_root,'2026-10-01_14-01-43-Jump-a'),
      'Sit':(remote_root,'2026-10-01_16-54-27-Sit-a'),
      'Pick':(remote_root,'2026-10-01_19-59-36-Pick-a'),
      'Bimanual-Pick':(remote_root,'2026-10-02_00-21-24-Bimanual-Pick-a')}
sources={n:library+'/'+n for n in ['algorithms/ppo.py','storage/rollout_storage.py','runners/on_policy_runner.py','modules/actor_critic.py']}
sources['exporter.py']=remote_root+'/third_party/DreamControl/Training/source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py'
sources['play_eval.py']=remote_root+'/third_party/DreamControl/Training/scripts/reinforcement_learning/rsl_rl/play_eval.py'
remote_code="""import hashlib,json,pathlib
sources=CONFIG['sources'];runs=CONFIG['runs'];root=CONFIG['root']
out={'sources':{},'exports':{}}
for name,p in sources.items():
 b=pathlib.Path(p).read_bytes();out['sources'][name]={'path':p,'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
for task,(r,run) in runs.items():
 base=pathlib.Path(r)/'third_party/DreamControl/Training/logs/rsl_rl/g1'/run
 item={'run':run,'files':{}}
 for name in ['model_1999.pt','exported/policy.pt','exported/policy.onnx']:
  p=base/name
  if p.exists():
   h=hashlib.sha256()
   with p.open('rb') as f:
    for block in iter(lambda:f.read(1048576),b''):h.update(block)
   item['files'][name]={'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
 p=pathlib.Path(root)/'artifacts/sim2sim_mujoco/interfaces'/f'{task}_interface.json'
 m=json.loads(p.read_text());item['interface']={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'obs_dim':m['snapshots'][0]['obs_dim'],'obs_terms':m['obs_terms'],'num_actions':m['num_actions'],'reference_count':m.get('reference_count',m.get('total_motions')),'policy_sha256':m['policy_sha256'],'hand_actions':m.get('hand_actions',{})}
 out['exports'][task]=item
print(json.dumps(out))
"""
remote_code='CONFIG='+repr({'sources':sources,'runs':runs,'root':remote_root})+'\n'+remote_code
cmd=['ssh','-b','10.147.17.250','-i','C:/Users/Pende/.ssh/id_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=15','ouyanghao@10.147.17.32','python3 -']
proc=subprocess.run(cmd,input=remote_code,capture_output=True,text=True,encoding='utf-8',timeout=90)
if proc.returncode:raise RuntimeError(proc.stderr)
evidence=json.loads(proc.stdout)
proof=ROOT/'论文复现汇报资料_20261002/PPO原理与策略导出'
proof.mkdir(parents=True,exist_ok=True)
for name,entry in evidence['sources'].items():
 p=proof/'源码核验'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(entry.pop('text'),encoding='utf-8',newline='\n')
evidence['collected_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(OUT/'source_and_policy_format_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
analysis={'aggregation':'Arithmetic mean of logger scalars; first/last 100 PPO iterations. Rolling best: largest 100-iteration mean, Punch windows do not cross resume1600. No statistical significance or full physics acceptance implied.','tasks':[]}
tags=['Train/mean_reward','Train/mean_episode_length','Loss/value_function','Loss/surrogate','Loss/entropy','Policy/mean_noise_std','Loss/learning_rate']
for task in runs:
 p=ROOT/'PPO训练曲线'/task/'iteration_scalars.csv'
 rows=list(csv.DictReader(p.open(encoding='utf-8-sig')));groups={}
 for r in rows:groups.setdefault(r['tag'],{})[int(r['iteration'])]=float(r['value'])
 # Actual entropy tag is verified against exported data, not guessed.
 if 'Loss/entropy' not in groups and 'Policy/entropy' in groups:entropy='Policy/entropy'
 else:entropy='Loss/entropy'
 selected=[entropy if t=='Loss/entropy' else t for t in tags]
 item={'task':task,'csv_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'metrics':{},'missing_stability_diagnostics':[]}
 for tag in selected:
  if tag not in groups:continue
  g=groups[tag];assert list(sorted(g))==list(range(2000)),(task,tag)
  a=np.array([g[i] for i in range(2000)]);assert np.isfinite(a).all()
  item['metrics'][tag]={'first100_mean':float(a[:100].mean()),'last100_mean':float(a[-100:].mean()),'last100_std_population':float(a[-100:].std()),'windows100':{str(s):float(a[s:s+100].mean()) for s in [0,400,900,1400,1500,1600,1900]}}
  if tag=='Train/mean_reward':
   windows=[(float(a[s:s+100].mean()),s) for s in range(1901) if not(task=='Punch' and s<1600<s+100)]
   best,start=max(windows)
   item['reward_best100']={'mean':best,'start':start,'end':start+99,'last_minus_best':float(a[-100:].mean()-best)}
 for term in ['kl','clip_fraction','explained_variance','grad_norm']:
  if not any(term in t.lower() for t in groups):item['missing_stability_diagnostics'].append(term)
 item['available_tags']=list(sorted(groups));analysis['tasks'].append(item)
(OUT/'six_task_curve_analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2),encoding='utf-8')
import shutil
for p in OUT.glob('*.json'):shutil.copy2(p,proof/p.name)
print(json.dumps({'obs_actions':{k:[v['interface']['obs_dim'],v['interface']['num_actions']] for k,v in evidence['exports'].items()},'metric_tags':list(analysis['tasks'][0]['metrics']),'proof':str(proof)},ensure_ascii=False))

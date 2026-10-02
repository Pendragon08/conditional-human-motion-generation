"""Archive already completed Ground-Pick-Top; no RL/evaluation launch."""
import json, subprocess, hashlib, shutil, datetime
from pathlib import Path
REPO=Path(r'D:\具身智能\conditional-human-motion-generation')
ROOT=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
REMOTE='/data/code/ouyanghao/oyh/conditional-human-motion-generation'
HOST='ouyanghao@10.147.17.32'
SSH=['ssh','-b','10.147.17.250','-i','C:/Users/Pende/.ssh/id_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST]
SCP=['scp','-o','BindAddress=10.147.17.250','-i','C:/Users/Pende/.ssh/id_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=15']
code="""import pathlib,json,hashlib,subprocess,re
r=pathlib.Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
p=r/'results/r3_dreamcontrol_official_Ground-Pick-Top_8192_eval.json';a=json.loads(p.read_text())
assert a['train_exit']==a['eval_exit']==0 and a['iterations']==2000 and a['eval_frames']==500 and a['num_envs']==8192 and a['eval_envs']==1000 and a['seed']==42
checks={}
for key in ['checkpoint','video','train_log','eval_log']:
 q=pathlib.Path(a[key]);q.relative_to(r);assert q.is_file()
 checks[key]={'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}
assert checks['checkpoint']['sha256']==a['checkpoint_sha256'] and checks['video']['sha256']==a['video_sha256']
assert re.search(r'Learning iteration\\s+1999/2000',pathlib.Path(a['train_log']).read_text(errors='replace'))
rates=re.findall(r'Success rate:\\s*([\\d.]+)\\s*%',pathlib.Path(a['eval_log']).read_text());assert float(rates[-1])==a['success_percent']
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate,duration','-show_entries','format=duration','-of','json',a['video']],text=True));v=probe['streams'][0]
assert int(v['nb_frames'])==501 and float(probe['format']['duration'])==10.02 and v['r_frame_rate']=='50/1'
cfg=pathlib.Path(a['checkpoint']).parent/'params'
configs={q.name:{'path':str(q),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in cfg.glob('*.yaml')}
print(json.dumps({'result':a,'checks':checks,'video_probe':probe,'configs':configs,'verified_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'eval_io':'official default Fabric path; no retry/runtime adaptation used','reference_count':1,'no_retraining':True,'no_new_evaluation':True}))
"""
p=subprocess.run(SSH+['python3 -'],input=code,capture_output=True,text=True,encoding='utf-8',timeout=90)
if p.returncode:raise RuntimeError(p.stderr)
w=json.loads(p.stdout);r=w['result']
for rel in ['results/completed_task_receipts','results/configs/DreamControl_Ground-Pick-Top','artifacts/archived_logs/Ground-Pick-Top_20261002']:(REPO/rel).mkdir(parents=True,exist_ok=True)
def copy(remote,local):
 local.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(SCP+[HOST+':'+remote,str(local)],check=True,timeout=90,capture_output=True)
 return hashlib.sha256(local.read_bytes()).hexdigest()
result_path=REPO/'results/r3_dreamcontrol_official_Ground-Pick-Top_8192_eval.json'
copy(REMOTE+'/results/'+result_path.name,result_path)
video=REPO/'media/r3_official_ground_pick_top_8192_eval.mp4'
assert copy(r['video'],video)==r['video_sha256']
for name,cfg in w['configs'].items():assert copy(cfg['path'],REPO/'results/configs/DreamControl_Ground-Pick-Top'/name)==cfg['sha256']
for key in ['train_log','eval_log']:
 local=REPO/'artifacts/archived_logs/Ground-Pick-Top_20261002'/Path(r[key]).name
 assert copy(r[key],local)==w['checks'][key]['sha256']
pack=ROOT/'论文复现汇报资料_20261002'
for dst in [pack/'media'/video.name,ROOT/'机器人执行视频/01_官方Isaac评估'/video.name]:dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(video,dst)
for src in [result_path,*list((REPO/'results/configs/DreamControl_Ground-Pick-Top').glob('*.yaml'))]:
 dst=pack/src.relative_to(REPO);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
receipt=REPO/'results/completed_task_receipts/Ground-Pick-Top_20261002.json';receipt.write_text(json.dumps(w,ensure_ascii=False,indent=2),encoding='utf-8')
(pack/'results/completed_task_receipts').mkdir(parents=True,exist_ok=True);shutil.copy2(receipt,pack/'results/completed_task_receipts'/receipt.name)
# Scalar-only postprocessing runs separately on CPU; no simulation app or queue is launched.
remote_script=REMOTE+'/scripts/export_training_report_20261002.py'
subprocess.run(SCP+[str(REPO/'scripts/export_training_report_20261002.py'),HOST+':'+remote_script],check=True,timeout=90,capture_output=True)
command=REMOTE+'/third_party/isaacsim-4.5.0/python.sh -u '+remote_script+' --task Ground-Pick Ground-Pick-Top --output-root '+REMOTE+'/artifacts/report_training_ground_completed_20261002'
export=subprocess.run(SSH+[command],capture_output=True,text=True,encoding='utf-8',timeout=180)
if export.returncode:raise RuntimeError(export.stderr[-4000:])
assert 'TRAINING_REPORT_EXPORT_COMPLETE 2' in export.stdout
print(export.stdout)
for task in ['Ground-Pick','Ground-Pick-Top']:
 subprocess.run(SCP+['-r',HOST+':'+REMOTE+'/artifacts/report_training_ground_completed_20261002/'+task,str(ROOT/'PPO训练曲线')],check=True,timeout=180,capture_output=True)
print(json.dumps({'task':'Ground-Pick-Top','success_percent':r['success_percent'],'video':str(video),'verified_model_sha256':r['checkpoint_sha256'],'logger_tasks':['Ground-Pick','Ground-Pick-Top']},ensure_ascii=False))

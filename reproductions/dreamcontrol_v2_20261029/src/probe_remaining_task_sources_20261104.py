import json,pathlib,subprocess,shutil
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage2_prior29_20261103';S=R/'stage3_pick29_20261103'
for p in [D/'config.json',D/'task_coverage_recheck_20261104/caption_retrieval_all_captions_v2.json',S/'reports/pick_table_bounded_0_v1/robot29.npz']:
 if p.suffix=='.json':
  j=json.loads(p.read_text());print('FILE',str(p));print(json.dumps(j,ensure_ascii=False)[:10500])
 else:print('INITIAL',p.exists(),p.stat().st_size)
print('GPU',subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True))
j=json.loads((S/'runs/pick500_8912_3000_v1/status.json').read_text());print('PPO',j['status'],j['policy_updates'])
print('DISK',shutil.disk_usage('/data').free)
print('SOURCES',subprocess.check_output(['rg','-n','--glob','*.py','left_wrist|orientation|wrist_roll|target_points',str(S/'src/stage3_pick500_postopt_follow_v3.py')],text=True)[-9000:])

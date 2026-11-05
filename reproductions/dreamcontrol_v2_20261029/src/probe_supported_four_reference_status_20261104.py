import json,pathlib,subprocess,hashlib,shutil
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage3_supermimic29_20261103';Q=R/'stage3_skill_sequence_20261103';O=R/'stage3_pick29_20261103/reference_scale500_20261104'
for name in ['sequence.json','progress.json']:
 p=Q/name;j=json.loads(p.read_text());print(name,json.dumps(j,ensure_ascii=False)[-19000:])
for p in [O/'final_library_audit.json',D/'active_native_universal_run.json',D/'native_universal_20261104/status.json',R/'stage3_pick29_20261103/runs/pick500_8912_3000_v1/status.json']:
 j=json.loads(p.read_text());print('STATUS',str(p),json.dumps(j,ensure_ascii=False)[-4500:])
print('STAGE_DIRS',[p.name for p in R.iterdir() if p.is_dir() and any(k in p.name.lower() for k in ['stage3','pour','box'])])
for root in [Q,D]:
 for p in root.glob('*'):
  if p.is_dir() and any(k in p.name.lower() for k in ['pour','box','reference']):print('CANDIDATE_DIR',str(p))
print('OWN_RUNNING',subprocess.check_output(['pgrep','-af','dreamcontrol-v2-reproduction-20261029.*(native_tracker|sample.*pour|box|pick500_ppo)'],text=True))
print('DISK',shutil.disk_usage('/data').free)

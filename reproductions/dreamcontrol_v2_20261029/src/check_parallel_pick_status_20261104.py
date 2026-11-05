import json,shutil,os,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';D=R/'stage3_supermimic29_20261103';W=D/'parallel_pick500_20261104'
for p in [W/'controller_v1/status.json',W/'capacity8912_v1/status.json',D/'runs/pick500_8912_72000_parallel_v1/status.json']:
    if p.exists():
        v=json.loads(p.read_text());print(str(p),json.dumps({k:v[k] for k in ['status','supermimic_status','supermimic_error','supermimic_phase','supermimic_formal_started','required_start_free_bytes','free_bytes','num_envs','progress','counters','latest_checkpoint','replay_bytes','elapsed_seconds'] if k in v},indent=2))
v=json.loads((W/'preflight.json').read_text());large=[]
for line in v['project_sizes']:
    a,b=line.split('\t',1)
    if int(a)>100*1024**2:large.append((int(a),b))
print('large_project_dirs',json.dumps(sorted(large,reverse=True)[:30],indent=2))
print('own_cache_candidates',subprocess.check_output(['find',str(R),'-maxdepth','3','-iname','*cache*','-type','d'],universal_newlines=True)[:5000])
print('free_bytes',shutil.disk_usage('/data').free)

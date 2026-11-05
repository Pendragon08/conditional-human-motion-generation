import json,pathlib,shutil
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');N=R/'stage3_supermimic29_20261103/native_universal_20261104'
for s in ['continuation_status.json','capacity_8912_v1_guard.json','benchmark_8912_v1_guard.json','benchmark_8912_v1/status.json']:
 p=N/s
 if p.exists():
  j=json.loads(p.read_text());print(s,json.dumps({k:j[k] for k in ['status','pid','phase','child_pid','error','resources','aggregate_resources','elapsed_seconds','optimizer_updates','checkpoint','generated_inductor_kernels'] if k in j}))
p=N/'benchmark_8912_v1/logs/train_log.csv'
if p.exists(): print('METRICS','\n'.join(p.read_text().splitlines()[-5:]))
p=N/'benchmark_8912_v1.log'
if p.exists(): print('LOG','\n'.join(p.read_text().splitlines()[-3:]))
p=R/'stage3_pick29_20261103/runs/pick500_8912_3000_v1/status.json';j=json.loads(p.read_text());print('PPO',j['status'],j['policy_updates'])
print('DISK',shutil.disk_usage('/data').free)

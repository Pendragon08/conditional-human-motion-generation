import json,pathlib,shutil
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';W=B/'bulk_v1'
p=W/'controller_status.json';j=json.loads(p.read_text());print(json.dumps({k:j[k] for k in ['status','pid','child_pid','active_task','sampled_new_candidates','native_admitted_counts','cohorts_completed','error','elapsed_seconds'] if k in j}))
for name in ['raw_audit.json','postopt/status.json','robot_audit.json','native/report.json','native_audit.json','summary.json']:
 p=W/'cohorts/pour_0001'/name
 if p.exists():
  j=json.loads(p.read_text());print(name,json.dumps({k:j[k] for k in ['status','completed','kinematic_passed','candidates','admitted_references','unique_admitted','rejected_candidates','error'] if k in j}))
p=R/'stage3_pick29_20261103/runs/pick500_8912_3000_v1/status.json';j=json.loads(p.read_text());print('PPO',j['status'],j['policy_updates']);print('DISK',shutil.disk_usage('/data').free)

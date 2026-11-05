import json,pathlib,shutil
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';D=R/'stage3_supermimic29_20261103'
for root in ['/data','/home/ouyanghao']:
 u=shutil.disk_usage(root);print('disk',root,u.free,u.total)
for path in sorted((B/'bulk_v1/cohorts').iterdir()):
 sizes={}
 for f in path.rglob('*'):
  if f.is_file() and not f.is_symlink():sizes[f.suffix]=sizes.get(f.suffix,0)+f.stat().st_size
 print('cohort',path.name,sizes)
for folder in [D/'runs/pick500_8912_72000_parallel_v2',D/'native_universal_20261104']:
 for p in folder.rglob('*.pt'):
  print('inactive_model_candidate',str(p),p.stat().st_size)
print('library_status',json.loads((B/'bulk_v1/controller_status.json').read_text())['native_admitted_counts'])
print('mixed_followup',json.loads((B/'mixed2000_completion_v1/status.json').read_text()))
print('log_archiver',json.loads((B/'bulk_v1/log_archive_status.json').read_text()))

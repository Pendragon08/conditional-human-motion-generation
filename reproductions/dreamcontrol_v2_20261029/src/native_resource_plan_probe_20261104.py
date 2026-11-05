import pathlib,shutil,json,os
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage3_supermimic29_20261103';N=D/'native_universal_20261104'
for path in ['/data','/home/ouyanghao','/dev/shm']:print(path,shutil.disk_usage(path)._asdict())
for p in [D/'runs/pick500_8912_72000_parallel_v2/resume_early_96.pt',D/'runs/pick500_8912_72000_parallel_v2/resume_failure_3424.pt',N/'smoke_author_v1/logs/last.pt',N/'benchmark_8912_v1/logs/ckpt_000010.pt']:print(str(p),p.stat().st_size)
print('SPEC',json.loads((N/'environment_spec.json').read_text())['spec'])

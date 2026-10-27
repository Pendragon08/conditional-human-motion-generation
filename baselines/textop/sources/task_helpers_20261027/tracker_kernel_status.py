import pathlib,json,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');p=r/'tracker_validation/author_v1/receipt.json'
print('RECEIPT',p.read_text() if p.exists() else 'NOT_WRITTEN')
log=r/'logs/tracker_kernel_author_v1.log'
if log.exists():
 with log.open('rb') as f:f.seek(max(0,log.stat().st_size-4500));print('LOG_TAIL',f.read().decode(errors='replace'))
print('GPU',subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader'],text=True))

import pathlib,json,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
for name in ['receipts/tracker_kernel_author_v2_exit.json','tracker_validation/author_v2/receipt.json','logs/tracker_supervisor_author_v2.log']:
 p=r/name
 if p.exists():print(name,p.read_text(errors='replace')[-3500:])
print('GPU',subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader'],text=True))

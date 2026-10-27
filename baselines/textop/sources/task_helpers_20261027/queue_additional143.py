import pathlib,subprocess,json,os
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');receipt=r/'receipts/additional143_queue.json'
if receipt.exists():raise FileExistsError('Existing queue preserved')
p=subprocess.Popen(['python3',str(r/'scripts/continue_available_raw.py')],stdout=open(r/'logs/additional143_queue.log','ab'),stderr=subprocess.STDOUT,start_new_session=True)
receipt.write_text(json.dumps({'pid':p.pid,'wait_for':'available500_v1 completion','next':'143 additional raw inputs, CPU4; no training'},indent=2));print(receipt.read_text())

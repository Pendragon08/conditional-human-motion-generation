import sys,pathlib,json,subprocess
TASK=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parent/'paper_faithful_reproduction_20261022'))
from session import REPO,remote
BASE=REPO/'baselines/textop';ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
record={'agent':'/root/textop_environment_doc_witness_v3','status':'pass','exit_code':0,'spec_hash':'9d7f370d','invocation':'verbatim PowerShell SSH command from corrected ENVIRONMENT.md','observed':{'sentinel':'TEXTOP_KERNEL_WITNESS','seed':0,'shape':[2,8,57],'device':'NVIDIA RTX A6000','python':'3.8.20','torch':'1.13.1+cu117','loss':1.302795648574829,'finite_gradients':True,'configured_cuda_fraction':.65,'params':42754233},'divergences':[],'limits':'Synthetic numerical witness only; training, source non-mutation, and dataset validation are separate evidence.'}
p=BASE/'results/doc_witness_pass.json';p.write_text(json.dumps(record,indent=2))
with (BASE/'SERVER_NOTES.md').open('a') as f:f.write('\nvalidated: spec9d7f370d fresh independent doc witness_v3 passed verbatim SSH invocation after corrected training-observability documentation, exit0, no divergence. Shared environment unchanged; attempt1 preserved, attempt2 uses official skip behavior and separate counters.\n')
subprocess.run(['scp','-i','C:/Users/Pende/.ssh/id_ed25519',str(p),'ouyanghao@10.147.17.32:'+ROOT+'/receipts/doc_witness_pass.json'],check=True)
print(remote(f'''
import pathlib,subprocess,json,os
r=pathlib.Path('{ROOT}')
assert not (r/'receipts/launcher.lock').exists()
p=subprocess.Popen(['python3',str(r/'scripts/launch_training.py')],cwd=r,stdout=open(r/'logs/gate.log','ab'),stderr=subprocess.STDOUT,start_new_session=True)
(r/'receipts/launcher_pid.json').write_text(json.dumps({{'pid':p.pid}}))
print('LAUNCH_GATE_PID',p.pid)
print('AVAILABLE_MEMORY',open('/proc/meminfo').read().split('MemAvailable:')[1].splitlines()[0])
'''))

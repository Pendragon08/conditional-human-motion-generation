"""Finite continuation: publish the terminal result of one authorized download."""
import json,os,pathlib,subprocess,sys,time
from session import REPO
BASE=REPO/'baselines/textop';HERE=pathlib.Path(__file__).resolve().parent
STATE=BASE/'results/data_preparation_20261025/cmu_resume.json'
sys.stdout.reconfigure(encoding='utf-8')
for _ in range(2880):
    try:status=json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}
    except json.JSONDecodeError:time.sleep(1);continue
    if status.get('status') in ['LOCAL_ORIGINAL_ARCHIVE_VERIFIED','STOPPED_WITH_ERROR_PARTIAL_RETAINED']:
        env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
        for name in ['collect_data_preparation.py','collect_storage_cleanup.py','publish.py']:
            proc=subprocess.run([sys.executable,str(HERE/name)],env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
            print(proc.stdout[-1800:],flush=True)
            if proc.returncode:raise RuntimeError(name+': '+proc.stderr[-1800:])
        print('CMU_TERMINAL_RESULT_REPORTED',flush=True);break
    time.sleep(30)
else:raise TimeoutError('24h continuation limit reached; download worker and all data retained')

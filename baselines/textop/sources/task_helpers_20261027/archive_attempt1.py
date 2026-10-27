import sys,pathlib,json,shutil,subprocess
TASK=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parent/'paper_faithful_reproduction_20261022'))
from session import remote,REPO
BASE=REPO/'baselines/textop';LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024');ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
a=BASE/'results/attempt1';a.mkdir(exist_ok=False)
for n in ['latest_status.json','training_progress.json','training_launch.json','training_exit.json','numerical_failure.json','dataset_actual_counts.json']:
    p=BASE/'results'/n
    if p.exists():shutil.copy2(p,a/n)
shutil.copy2(BASE/'reports/启动报告_20261024.md',a/'report_at_failure.md')
if (BASE/'results/logs').exists():(BASE/'results/logs').rename(a/'logs')
print(remote(r'''
import pathlib,json,tarfile,hashlib,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
assert json.loads((r/'receipts/training_exit.json').read_text())['returncode']==1
for where,names in [('receipts',['training_progress.json','training_launch.json','training_exit.json','numerical_failure.json','training_failure.txt','launcher.lock','launcher_pid.json']),('logs',['mvae.log','mvae_metrics.jsonl','gate.log'])]:
    dst=r/where/'attempt1';dst.mkdir(exist_ok=False)
    for name in names:
        src=r/where/name
        if src.exists():src.rename(dst/name)
shutil.copy2(r/'scripts/train_official.py',r/'receipts/attempt1/train_official_v1.py')
archive=r/'receipts/attempt1_complete_records.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    t.add(r/'receipts/attempt1',arcname='receipts');t.add(r/'logs/attempt1',arcname='logs')
print(json.dumps({'path':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size}))
'''))
subprocess.run(['scp','-i','C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32:'+ROOT+'/receipts/attempt1_complete_records.tar.gz',str(a/'complete_records.tar.gz')],check=True)
shutil.copy2(a/'complete_records.tar.gz',LOCAL/'attempt1_complete_records.tar.gz')
# Current top-level failures remain available in attempt1; latest top-level state will describe attempt2.
for n in ['training_exit.json','numerical_failure.json','training_progress.json','training_launch.json']:
    p=BASE/'results'/n
    if p.exists():p.rename(a/('original_'+n))

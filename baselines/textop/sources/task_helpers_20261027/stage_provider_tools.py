import pathlib,sys,shutil
TASK=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parent/'paper_faithful_reproduction_20261022'))
from session import REPO
dst=REPO/'baselines/textop/provider_tools'
for name in ['collect.py','publish.py']:
    source=(TASK/name).read_text(encoding='utf-8')
    source=source.replace("sys.path.insert(0,str(TASK.parent/'paper_faithful_reproduction_20261022'))\n",'')
    (dst/name).write_text(source,encoding='utf-8')
(dst/'README.md').write_text('# Windows provider tools\n\nRun collect.py to save reports locally and copy server receipts, then publish.py to push only baselines/textop to its configured GitHub branch. These scripts use the current workstation/SSH provider paths from session.py. No passwords, tokens or private-key contents are stored.\n\nThe checkpoint sync worker uses ../configs/checkpoint-sync.local.json. Keep the workstation online for automatic backups. It never touches datasets or other experiment runs; only older checkpoint copies of the explicit run are eligible after fresh local SHA256 and GitHub digest checks, and a newer verified server checkpoint is retained.\n',encoding='utf-8')
print('PROVIDER_TOOLS_STAGED')

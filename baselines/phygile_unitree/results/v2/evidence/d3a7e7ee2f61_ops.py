"""Task-local SSH/file transport; no deletion or process interruption helpers."""
from pathlib import Path
import argparse,json,subprocess,sys

SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=4','-i','C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32']
def remote(script,python='python3',timeout=180):
    return subprocess.run(SSH+[python+' -'],input=script,capture_output=True,text=True,encoding='utf-8',timeout=timeout)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('script',type=Path);p.add_argument('--python',default='python3');p.add_argument('--output',type=Path);p.add_argument('--timeout',type=int,default=180);a=p.parse_args()
    r=remote(a.script.read_text(encoding='utf-8'),a.python,a.timeout)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    sys.stdout.buffer.write(r.stdout.encode('utf-8'));sys.stderr.buffer.write(r.stderr.encode('utf-8'));sys.exit(r.returncode)

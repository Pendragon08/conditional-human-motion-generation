"""Persist the actual child exit status, including failures before run creation."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
command=a.command[1:] if a.command[:1]==['--'] else a.command
assert command and not a.record.exists()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
started=now();child=subprocess.Popen(command)
receipt={'status':'RUNNING','started_utc':started,'pid':child.pid,'command':command}
def save():
 t=a.record.with_suffix('.partial');t.write_text(json.dumps(receipt,indent=2)+'\n');t.replace(a.record)
save();code=child.wait();receipt.update(status='EXITED',exit_code=code,finished_utc=now());save();sys.exit(code)

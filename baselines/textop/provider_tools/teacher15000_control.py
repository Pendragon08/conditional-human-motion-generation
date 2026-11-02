"""New15k control with unique client evidence; no blind repeat after unknown exit."""
import argparse,json,pathlib,hashlib,sys
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--action',choices=['observe','finish'],required=True);p.add_argument('--evidence',required=True);a=p.parse_args()
assert a.evidence.isidentifier() and a.evidence.isascii()
folder=B/'results/teacher15000_direct_strict_20261027/control';folder.mkdir(parents=True,exist_ok=True)
digest=hashlib.sha256((B/'configs/teacher15000-direct-evaluation-spec.json').read_bytes()).hexdigest()
code="import pathlib,hashlib,subprocess,sys\nr=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');p=r/'configs/teacher15000-direct-evaluation-spec.json'\nassert hashlib.sha256(p.read_bytes()).hexdigest()=="+repr(digest)+"\nv=subprocess.run([sys.executable,str(r/'scripts/teacher15000_evaluation_control.py'),'--action',"+repr(a.action)+"],capture_output=True,text=True,timeout=90)\nprint(v.stdout);print(v.stderr,file=sys.stderr);raise SystemExit(v.returncode)\n"
raw=recorded_remote(code,folder/(a.evidence+'.json'),timeout=110)
result=json.loads(raw);(folder/(a.evidence+'_result.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))

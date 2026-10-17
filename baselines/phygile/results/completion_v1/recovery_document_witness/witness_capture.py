from pathlib import Path
import datetime, hashlib, json, shlex, subprocess, sys, time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DOC = ROOT / 'docs/terminal_recovery_invocation.md'
HOST = 'ouyanghao@10.147.17.32'
REMOTE = '/data/code/ouyanghao/oyh/phygile-reproduction-20261017'
INVOKE = 'bash scripts/run_isaac.sh scripts/validate_terminal_boundary.py --name closure_recovery'
SSH = ['C:/Windows/System32/OpenSSH/ssh.exe', '-i', 'C:/Users/Pende/.ssh/id_ed25519', '-p', '22', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST]

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def capture(name, command):
    record = {'started_utc': now(), 'argv': SSH + [command], 'remote_command': command}
    (OUT / (name + '.command.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
    with (OUT / (name + '.stdout.log')).open('wb') as stdout, (OUT / (name + '.stderr.log')).open('wb') as stderr:
        proc = subprocess.Popen(SSH + [command], stdout=stdout, stderr=stderr)
        (OUT / (name + '.pid')).write_text(str(proc.pid), encoding='ascii')
        code = proc.wait()
    record.update(ended_utc=now(), exit_code=code)
    (OUT / (name + '.result.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)
    return code

stage = sys.argv[1]
if stage == 'preflight':
    (OUT / 'invocation_document.snapshot.md').write_bytes(DOC.read_bytes())
    (OUT / 'document_sha256.txt').write_text(hashlib.sha256(DOC.read_bytes()).hexdigest() + '\n', encoding='ascii')
    cmd = "cd " + REMOTE + " && hostname && date -u +%FT%TZ && nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader && printf '\\nTASK_PROCESSES\\n' && (pgrep -af '[v]alidate_terminal_boundary.py|[r]un_generated_closed_loop.py' || true) && printf '\\nOUTPUT_DIRECTORY_EXISTS\\n' && if test -e results/completion_v1/closure_recovery; then printf 'yes\\n'; else printf 'no\\n'; fi"
    raise SystemExit(capture(stage, cmd))
elif stage == 'run':
    if (OUT / 'run.command.json').exists():
        raise RuntimeError('Refusing a second documented invocation')
    raise SystemExit(capture(stage, 'cd ' + REMOTE + ' && ' + INVOKE))
elif stage == 'monitor':
    cmd = 'cd ' + REMOTE + " && date -u +%FT%TZ && tail -n 35 results/completion_v1/closure_recovery/run/stdout_stderr.log && printf '\\nRESULT_FILES\\n' && find results/completion_v1/closure_recovery -type f -printf '%P %s bytes\\n'"
    raise SystemExit(capture(stage + '_' + str(int(time.time())), cmd))
elif stage == 'collect':
    script = '''from pathlib import Path
import hashlib,json
root=Path('.')
base=root/'results/completion_v1/closure_recovery'
records=[]
for p in sorted(base.rglob('*')):
 if p.is_file():
  records.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
extras={}
for s in ['docs/terminal_recovery_invocation.md','docs/environment.md','configs/env-spec.json','configs/nominal_standing_reference.json','scripts/run_isaac.sh','scripts/validate_terminal_boundary.py','scripts/run_generated_closed_loop.py','scripts/replay_physics_worker.py','reimplementation/replay_prefix_adapter.py']:
 p=root/s
 extras[s]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()}
print(json.dumps({'remote_cwd':str(root.resolve()),'files':records,'inputs':extras},indent=2))
'''
    code=capture('remote_manifest', 'cd ' + REMOTE + ' && python3 -c ' + shlex.quote(script))
    if code:
        raise SystemExit(code)
    dest=OUT/'raw_result'
    if dest.exists():
        raise RuntimeError('Refusing to overwrite raw evidence copy')
    argv=['C:/Windows/System32/OpenSSH/scp.exe','-i','C:/Users/Pende/.ssh/id_ed25519','-P','22','-o','BatchMode=yes','-o','ConnectTimeout=15','-r',HOST+':'+REMOTE+'/results/completion_v1/closure_recovery',str(dest)]
    record={'started_utc':now(),'argv':argv}
    proc=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (OUT/'copy.stdout.log').write_bytes(proc.stdout)
    (OUT/'copy.stderr.log').write_bytes(proc.stderr)
    record.update(exit_code=proc.returncode,ended_utc=now())
    (OUT/'copy.result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)
    raise SystemExit(proc.returncode)
else:
    raise ValueError(stage)

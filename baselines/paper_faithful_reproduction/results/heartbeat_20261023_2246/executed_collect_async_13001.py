from session import *
import tarfile
script=f'''from pathlib import Path
import tarfile,json,hashlib
root=Path({REMOTE!r}); run=root/'reactive_async_v8_13001'
receipt=json.loads((run/'receipt.json').read_text()); assert receipt['status']=='ASYNC_FEEDBACK_APPLIED'
assert json.loads((root/'async_v8_13001_exit.json').read_text())['exit_code']==0
out=root/'async_evidence_13001.tar.gz';assert not out.exists()
with tarfile.open(out,'w:gz') as t:
 for p in run.rglob('*'):
  if p.is_file():t.add(p,arcname=str(p.relative_to(root)))
 for name in ['async_v8_13001.stdout.txt','async_v8_13001.stderr.txt','async_v8_13001_exit.json','async_launch_v8_13001.json','code_v4/run_reactive_async.py','training_fp64_sdpa_v4/checkpoint_ready_000013000.json']:
  t.add(root/name,arcname=name)
print(json.dumps({{'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}}))
'''
info=json.loads(remote(script));archive=LOCAL/'async_evidence_13001.tar.gz'
with archive.open('xb') as output:
    r=subprocess.run(SSH+['cat -- '+REMOTE+'/async_evidence_13001.tar.gz'],stdout=output,stderr=subprocess.PIPE,timeout=120)
assert r.returncode==0,r.stderr
assert archive.stat().st_size==info['bytes'] and sha(archive.read_bytes())==info['sha256']
dest=LOCAL/'reactive_async_records_13001';dest.mkdir(exist_ok=False)
with tarfile.open(archive) as t:
    for m in t:
        assert (dest/m.name).resolve().is_relative_to(dest.resolve())
    t.extractall(dest,filter='data')
public=BASE/'results/reactive_async_v8_13001';public.mkdir(exist_ok=False)
run=dest/'reactive_async_v8_13001'
for p in run.rglob('*.json'):
    q=public/p.relative_to(run);q.parent.mkdir(exist_ok=True,parents=True);q.write_bytes(p.read_bytes())
dump(public/'archive_receipt.json',info)
receipt=json.loads((run/'receipt.json').read_text(encoding='utf-8'))
print(json.dumps({k:v for k,v in receipt.items() if k not in ['source_hashes','planner_calls','events']}))

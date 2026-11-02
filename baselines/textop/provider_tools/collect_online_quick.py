"""Collect only closed online runs; raw evidence stays private on server and D."""
import argparse,hashlib,json,pathlib,subprocess
from session import remote,SSH
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();assert a.run.isidentifier()
B=pathlib.Path(__file__).resolve().parents[1];D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
code=r'''
import pathlib,json,hashlib,tarfile,shutil,statistics,os
os.sched_setaffinity(0,{0,1,2,3});os.nice(10)
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');run=RUN;o=r/'online_private'/run;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
outer=r/'receipts'/('online_'+run+'_outer_exit.json');ex=json.loads(outer.read_text());j=json.loads((o/'pipeline.json').read_text());assert 'actual_generator_returncode' in j
sr=json.loads((o/'sdk/receipt.json').read_text()) if (o/'sdk/receipt.json').exists() else {};gr=json.loads((o/'generator_receipt.json').read_text())
backup=r/'private_backups'/('online_'+run);backup.mkdir(mode=0o700,exist_ok=True);archive=backup/'evidence.private.tar.gz'
if not archive.exists():
 assert shutil.disk_usage(r).free>3*1024**3
 with tarfile.open(archive,'x:gz') as t:t.add(o,arcname=run);t.add(outer,arcname='outer_exit.json')
with tarfile.open(archive,'r:gz') as t:
 members=t.getmembers()
 for m in members:
  if m.isfile():
   stream=t.extractfile(m)
   while stream.read(1024**2):pass
assert archive.stat().st_size<150*1024**2
rows=[json.loads(x) for x in (o/'sdk/timing.private.jsonl').read_text().splitlines()] if (o/'sdk/timing.private.jsonl').exists() else []
durations={}
for left,right in [('actual_start','physics_done'),('physics_done','tick_done'),('actual_start','poll_done'),('poll_done','policy_done'),('policy_done','physics_done')]:
 if rows and left in rows[0] and right in rows[0]:
  v=sorted(x[right]-x[left] for x in rows);durations[left+'_to_'+right]={'mean':statistics.mean(v),'p50':v[len(v)//2],'p95':v[int(.95*len(v))],'max':max(v)}
summary={'run':run,'status':j['status'],'outer_returncode':ex['actual_returncode'],'spec_sha256':j['spec_sha256'],
 'physical_failure':sr.get('physical_failure'),'control_calls':sr.get('actual_control_calls',len(rows)),'warmup_control_calls':sr.get('warmup_control_calls',0),
 'hot_command_consumed':sr.get('switched_target_consumed') is not None,'full_bounded_duration':sr.get('full_bounded_duration',False),'deadline_misses':sr.get('deadline_misses'),
 'all_deadlines_passed':sr.get('actual_50hz_deadlines_passed',False),'max_deadline_miss_seconds':sr.get('max_deadline_miss_seconds'),'wall_elapsed_seconds':sr.get('wall_elapsed_seconds'),'control_loop_elapsed_seconds':sr.get('control_loop_elapsed_seconds'),
 'stream_underruns':sr.get('stream_underruns'),'durations':durations,'error':sr.get('error'),'generator_updates':gr['optimizer_updates'],'sdk_updates':sr.get('optimizer_updates',0),
 'semantic_quality_accepted':False,'full_paper_training_started':False,'receipt_sha256':sha(o/'pipeline.json'),'sdk_receipt_sha256':sha(o/'sdk/receipt.json') if sr else None,
 'generator_receipt_sha256':sha(o/'generator_receipt.json'),'outer_sha256':sha(outer),'archive':{'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'members':len(members)}}
print(json.dumps(summary))
'''.replace('RUN',repr(a.run))
j=json.loads(remote(code,timeout=120));target=D/'diagnostics/online_quick_20261026'/a.run;target.mkdir(parents=True,exist_ok=True);dest=target/'evidence.private.tar.gz'
if not dest.exists():
    scp=['scp']+SSH[1:-1]+[SSH[-1]+':'+j['archive']['path'],str(dest)]
    subprocess.run(scp,check=True,timeout=180,capture_output=True)
assert dest.stat().st_size==j['archive']['bytes'] and hashlib.sha256(dest.read_bytes()).hexdigest()==j['archive']['sha256']
j['private_backup_verified']=True;j['local_private_archive']=str(dest)
(target/'summary.json').write_text(json.dumps(j,indent=2),encoding='utf-8');o=B/'results/online_quick_20261026'/a.run;o.mkdir(exist_ok=True)
(o/'summary.json').write_text(json.dumps(j,indent=2),encoding='utf-8');print(json.dumps(j))

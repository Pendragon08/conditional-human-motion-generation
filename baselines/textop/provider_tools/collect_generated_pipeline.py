"""Collect actual generated-ref SDK stages and one closed private trace archive."""
import argparse,hashlib,json,pathlib,shutil,subprocess
from session import remote
B=pathlib.Path(__file__).resolve().parents[1];D=pathlib.Path('D:/具身智能/TextOp复现_20261024');p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--backup-closed',action='store_true');a=p.parse_args();assert a.run.isidentifier() and a.run.isascii()
script=r'''
import hashlib,json,pathlib,os,tarfile
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');run=RUN;root=r/'generated_execution_private'/run
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();result={'run':run,'stages':{},'pipeline':None,'actual_supervisor_exit':None,'archive':None}
for stage in ['export','replay']:
 rp=root/stage/'receipt.json';ep=r/'receipts'/('generated_pipeline_'+run+'_'+stage+'_exit.json')
 if rp.exists():
  entry={'receipt':json.loads(rp.read_text()),'receipt_sha256':sha(rp),'exit':None};result['stages'][stage]=entry
  if ep.exists():
   ex=json.loads(ep.read_text());assert ex['evaluation_receipt_sha256']==sha(rp)
   assert sha(r/'receipts'/('generated_pipeline_'+run+'_'+stage+'_process_exit.json'))==ex['process_exit_sha256']
   assert sha(r/'logs'/('generated_pipeline_'+run+'_'+stage+'_resources.jsonl'))==ex['resources_sha256']
   entry.update(exit=ex,exit_sha256=sha(ep))
fp=root/'pipeline.json'
if fp.exists():result['pipeline']=json.loads(fp.read_text());result['pipeline_sha256']=sha(fp)
outer=r/'receipts'/(run+'_supervisor_execution.json')
if outer.exists():result['actual_supervisor_exit']=json.loads(outer.read_text())['exit_code']
if BACKUP and result['pipeline'] is not None:
 assert result['actual_supervisor_exit']==0 and result['pipeline']['output_writers_closed']
 assert all(result['stages'][s]['exit']['supervision_passed'] and result['stages'][s]['exit']['child_exit_code']==0 for s in ['export','replay'])
 export=result['stages']['export']['receipt'];replay=result['stages']['replay']['receipt'];assert sha(export['reference']['file'])==export['reference']['sha256']
 assert sha(root/'replay/executed_traces.private.json')==replay['private_trace_index_sha256']
 index=json.loads((root/'replay/executed_traces.private.json').read_text());assert len(index)==1
 trace=root/'replay/execution_private/trial_0';meta=json.loads((trace/'trace.private.json').read_text())
 assert sha(trace/'poses.private.bin')==meta['binary_sha256'] and sha(trace/'executed_poses.private.npz')==meta['npz_sha256']
 folder=r/'private_backups'/('generated_'+run);folder.mkdir(parents=True,mode=0o700,exist_ok=True);dest=folder/'trial.private.tar.gz';manifest=folder/'archive.private.json'
 files=[fp,root/'export/receipt.json',pathlib.Path(export['reference']['file']),root/'replay/receipt.json',root/'replay/executed_traces.private.json',trace/'trace.private.json',trace/'poses.private.bin',trace/'executed_poses.private.npz']
 if not dest.exists():
  assert not manifest.exists();os.sched_setaffinity(0,{0,1,2,3})
  with tarfile.open(dest,'x:gz') as tar:
   for f in files:tar.add(f,arcname=f.relative_to(root).as_posix(),recursive=False)
  with tarfile.open(dest,'r:gz') as tar:
   assert len(tar.getmembers())==len(files)
   for f in files:assert hashlib.sha256(tar.extractfile(f.relative_to(root).as_posix()).read()).hexdigest()==sha(f)
  manifest.write_text(json.dumps({'file':str(dest),'bytes':dest.stat().st_size,'sha256':sha(dest),'members':{f.relative_to(root).as_posix():sha(f) for f in files}},indent=2))
 archived=json.loads(manifest.read_text());assert archived['sha256']==sha(dest) and archived['members']=={f.relative_to(root).as_posix():sha(f) for f in files};result['archive']=archived
print(json.dumps(result))
'''.replace('RUN',repr(a.run)).replace('BACKUP',repr(a.backup_closed))
j=json.loads(remote(script,timeout=100));out=B/'results/generated_execution_20261026'/a.run;out.mkdir(parents=True,exist_ok=True)
for stage,e in j['stages'].items():
 folder=out/stage;folder.mkdir(exist_ok=True);(folder/'receipt.json').write_text(json.dumps(e['receipt'],indent=2),encoding='utf-8')
 if e['exit'] is not None:(folder/'exit.json').write_text(json.dumps(e['exit'],indent=2),encoding='utf-8')
if j['pipeline'] is not None:(out/'pipeline.json').write_text(json.dumps(j['pipeline'],indent=2),encoding='utf-8')
if j['archive']:
 archive=j['archive'];private=D/'datasets/generated_execution_private_20261026'/a.run;private.mkdir(parents=True,exist_ok=True);dest=private/'trial.private.tar.gz'
 if not dest.exists():
  assert shutil.disk_usage(private).free>archive['bytes']+1024**3
  z=subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','-i','C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32:'+archive['file'],str(dest)],capture_output=True,text=True,timeout=120);assert z.returncode==0,z.stderr
 assert dest.stat().st_size==archive['bytes'] and hashlib.sha256(dest.read_bytes()).hexdigest()==archive['sha256']
 (private/'archive.private.json').write_text(json.dumps(archive,indent=2),encoding='utf-8')
 (out/'backup.json').write_text(json.dumps({'bytes':archive['bytes'],'sha256':archive['sha256'],'private_two_location_sha_verified':True,'raw_arrays_publicly_uploaded':False},indent=2),encoding='utf-8')
shutil.copytree(out,D/'results/generated_execution_20261026'/a.run,dirs_exist_ok=True)
print(json.dumps({'run':a.run,'stages':{s:{'status':v['receipt']['status'],'trial':v['receipt'].get('trial'),'child_exit':v['exit']['child_exit_code'] if v['exit'] else None,'error':v['receipt'].get('error')} for s,v in j['stages'].items()},'pipeline_closed':j['pipeline'] is not None,'actual_supervisor_exit':j['actual_supervisor_exit'],'private_archive_backed_up':bool(j['archive'])}))

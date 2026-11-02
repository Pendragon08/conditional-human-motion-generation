"""Back up private corpus bytes; publish only aggregate real loader evidence."""
import argparse,hashlib,json,pathlib,shutil,subprocess
from session import remote
B=pathlib.Path(__file__).resolve().parents[1]
D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
assert a.run.isidentifier() and a.run.isascii()
script=r'''
import pathlib,json,hashlib,tarfile,os
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');run=RUN_LITERAL
os.sched_setaffinity(0,{0,1,2,3});sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=r/'quick_generator_20261026/corpus_v1';rp=r/'quick_generator_20261026/corpus_validation'/run/'receipt.json';ep=r/'receipts'/('quick69_corpus_'+run+'_exit.json')
rec=json.loads(rp.read_text());ex=json.loads(ep.read_text());assert rec['status']=='QUICK69_CORPUS_LOADER_WITNESS_PASSED' and rec['output_writers_closed'] is True
assert ex['supervision_passed'] is True and ex['child_exit_code']==0 and ex['child_reaped'] is True and ex['timed_out'] is False
assert sha(rp)==ex['evaluation_receipt_sha256']
for p,k in [(r/'receipts'/('quick69_corpus_'+run+'_process_exit.json'),'process_exit_sha256'),(r/'logs'/('quick69_corpus_'+run+'_resources.jsonl'),'resources_sha256')]:assert sha(p)==ex[k]
cp=out/'corpus_receipt.json';assert sha(cp)==rec['corpus_receipt_sha256'];meta=json.loads(cp.read_text())
for n,x in meta['artifacts'].items():assert sha(out/n)==x['sha256'] and (out/n).stat().st_size==x['bytes']
dest=r/'private_backups/quick69_corpus_v1';dest.mkdir(mode=0o700,exist_ok=True);archive=dest/'corpus.private.tar.gz';proof=dest/'backup.json'
if not archive.exists():
 assert not proof.exists();files=[out/n for n in meta['artifacts']]+[cp]
 assert all(x.is_file() and not x.is_symlink() for x in files)
 assert os.statvfs(dest).f_bavail*os.statvfs(dest).f_frsize>sum(x.stat().st_size for x in files)+2*1024**3
 with archive.open('xb') as target:
  with tarfile.open(fileobj=target,mode='w:gz') as t:
   for p in files:t.add(str(p),arcname=p.name,recursive=False)
 os.chmod(archive,0o600)
 with tarfile.open(archive,'r:gz') as t:
  assert len(t.getmembers())==len(files)
  for p in files:assert hashlib.sha256(t.extractfile(p.name).read()).hexdigest()==sha(p)
 saved={'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'path':str(archive),'corpus_receipt_sha256':sha(cp),'all_members_verified':True};proof.write_text(json.dumps(saved,indent=2))
else:
 saved=json.loads(proof.read_text());assert saved['corpus_receipt_sha256']==sha(cp) and saved['archive_sha256']==sha(archive)
print(json.dumps({'receipt':rec,'exit':ex,'backup':saved,'receipt_sha256':sha(rp),'exit_sha256':sha(ep)}))
'''.replace('RUN_LITERAL',repr(a.run))
j=json.loads(remote(script,timeout=90));meta=j['backup']
private=D/'datasets/quick69_corpus_private_20261026';private.mkdir(parents=True,exist_ok=True);target=private/'corpus.private.tar.gz'
if not target.exists():
    assert shutil.disk_usage(private).free>meta['bytes']+1024**3
    partial=target.with_suffix('.gz.partial');assert not partial.exists()
    p=subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','-i','C:/Users/Pende/.ssh/id_ed25519',
        'ouyanghao@10.147.17.32:'+meta['path'],str(partial)],capture_output=True,text=True,timeout=120)
    assert p.returncode==0,p.stderr
    assert partial.stat().st_size==meta['bytes'] and hashlib.sha256(partial.read_bytes()).hexdigest()==meta['archive_sha256'];partial.replace(target)
assert target.stat().st_size==meta['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==meta['archive_sha256']
(private/'backup.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
public=B/'results/quick69_corpus_20261026'/a.run;public.mkdir(parents=True,exist_ok=True)
for key in ['receipt','exit']:(public/(key+'.json')).write_text(json.dumps(j[key],indent=2),encoding='utf-8')
proof={'status':'PRIVATE_QUICK_CORPUS_AND_REAL_LOADER_EVIDENCE_BACKED_UP','run':a.run,
    'private_server_and_local_sha_verified':True,'archive_sha256':meta['archive_sha256'],'archive_bytes':meta['bytes'],
    'corpus_receipt_sha256':meta['corpus_receipt_sha256'],'witness_receipt_sha256':j['receipt_sha256'],'witness_exit_sha256':j['exit_sha256'],
    'private_motion_arrays_or_text_publicly_published':False,'quick_training_ready':False,'full_paper_dataset_accepted':False}
(public/'backup.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
local=D/'results/quick69_corpus_20261026'/a.run;local.mkdir(parents=True,exist_ok=True)
for p in public.iterdir():
    if p.is_file():shutil.copy2(p,local/p.name);assert p.read_bytes()==(local/p.name).read_bytes()
print(json.dumps(proof))

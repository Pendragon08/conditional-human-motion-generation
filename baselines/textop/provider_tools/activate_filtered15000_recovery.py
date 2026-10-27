"""Enable bounded private recovery only after author/fresh real corpus witnesses."""
import argparse,pathlib,json,hashlib,base64
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--document-summary',required=True);a=p.parse_args()
summary_path=pathlib.Path(a.document_summary).resolve();summary=json.loads(summary_path.read_text(encoding='utf-8'))
assert summary['status']=='FILTERED15000_CORPUS_FRESH_DOCUMENT_VERIFIED' and summary['run']=='filtered15000_corpus_document_v1'
assert summary['actual_exit_code']==0 and summary['findings']==[]
cp=B/'configs/filtered15000-corpus-spec.json';corpus_sha=hashlib.sha256(cp.read_bytes()).hexdigest()
assert summary['corpus_spec_sha256']==corpus_sha
sp=B/'configs/filtered15000-recovery-spec.json';old=sp.read_bytes();s=json.loads(old);assert not s['prepare_authorized']
witnesses=[]
for run,mode in [('filtered15000_corpus_author_v1','build'),('filtered15000_corpus_document_v1','verify')]:
 q=B/'results/filtered15000_handoff_20261027'/run/'progress.json';j=json.loads(q.read_text(encoding='utf-8'))
 assert j['kind']=='corpus' and j['closed_execution_verified'] and not j['workers']
 assert j['files']['exit']['mode']==mode and j['files']['exit']['spec_sha256']==corpus_sha
 witnesses.append(dict(run=run,mode=mode,receipt_sha256=j['sha256']['receipt'],exit_sha256=j['sha256']['exit']))
assert j['common_lock_free']
s.update(prepare_authorized=True,corpus_spec_sha256=corpus_sha,corpus_witnesses=witnesses,
 fresh_document_summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
 vae_spec_sha256=hashlib.sha256((B/'configs/filtered15000-vae-spec.json').read_bytes()).hexdigest())
new=(json.dumps(s,indent=2)+'\n').encode();expected=hashlib.sha256(old).hexdigest()
code=r'''
import pathlib,json,hashlib,base64,sys,os,fcntl,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
p=r/'configs/filtered15000-recovery-spec.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==OLD
data=base64.b64decode(NEW);s=json.loads(data);sys.path.insert(0,str(r/'scripts'))
assert hashlib.sha256((r/'configs/filtered15000-corpus-spec.json').read_bytes()).hexdigest()==s['corpus_spec_sha256']
for n,h in dict(s['bound_sources'],**{'scripts/prepare_filtered15000_recovery.py':s['executor_sha256'],'scripts/supervise_filtered15000_recovery.py':s['supervisor_sha256']}).items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
for w in s['corpus_witnesses']:
 rp=r/'filtered15000_20261027/corpus_runs'/w['run']/'receipt.json';prefix='filtered15000_corpus_'+w['run']
 ep=r/'receipts'/(prefix+'_exit.json');outer=r/'receipts'/(prefix+'_outer_exit.json')
 assert hashlib.sha256(rp.read_bytes()).hexdigest()==w['receipt_sha256'] and hashlib.sha256(ep.read_bytes()).hexdigest()==w['exit_sha256']
 rec=json.loads(rp.read_text());e=json.loads(ep.read_text());assert rec['status']=='FILTERED15000_CORPUS_LOADER_WITNESS_PASSED' and rec['output_writers_closed']
 assert e['mode']==w['mode'] and e['child_exit_code']==0 and e['child_reaped'] and e['supervision_passed'] and not e['timed_out']
 assert json.loads(outer.read_text())['actual_returncode']==0
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert shutil.disk_usage(r).free>=s['minimum_disk_free_bytes']
archive=r/'receipts/filtered15000_recovery_spec_before_enable.json'
with archive.open('xb') as f:f.write(p.read_bytes())
temp=p.with_name(p.name+'.enable.partial')
with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
temp.replace(p)
result=dict(status='PRIVATE_RECOVERY_PREPARATION_ENABLED_NOT_DISPATCHED',old_spec_sha256=OLD,new_spec_sha256=hashlib.sha256(data).hexdigest(),
 corpus_witnesses=s['corpus_witnesses'],fresh_document_summary_sha256=s['fresh_document_summary_sha256'],new_model_loads=0,new_gpu_contexts=0,new_jobs=0)
with (r/'receipts/filtered15000_recovery_activation.json').open('x') as f:json.dump(result,f,indent=2)
lock.close();print(json.dumps(result))
'''.replace('OLD',repr(expected)).replace('NEW',repr(base64.b64encode(new).decode()))
folder=B/'results/filtered15000_handoff_tools_20261027'
result=json.loads(recorded_remote(code,folder/'enable_recovery_v1_client.json',timeout=90));sp.write_bytes(new)
(folder/'recovery_activation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

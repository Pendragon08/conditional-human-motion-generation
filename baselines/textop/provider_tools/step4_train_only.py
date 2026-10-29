"""New explicit Stage4 original-TRAIN-only route; Stage3 remains disabled."""
import argparse
import ast
import base64
import hashlib
import json
import os
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
ROOT = '/data/code/ouyanghao/oyh/textop-reproduction-20261024'
O = B/'results/step4_train_only_20261029'
SOURCES = ['scripts/step4_train_only_protocol.py', 'scripts/export_step4_train_only_references.py',
    'scripts/supervise_step4_train_only_references.py']
BOUND = SOURCES + ['scripts/export_tracker_references.py', 'scripts/bounded_headless_close.py',
    'scripts/run_tracker.sh', 'scripts/contact_filter_resources.py', 'scripts/contact_process_supervision.py',
    'configs/tracker-env-spec.json']


def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def write(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.flush(); os.fsync(f.fileno())


def run_name(role): return 'user_step4_train_only_fk_'+role+'_v1'


def launch(role, evidence):
    out = O/evidence; out.mkdir(parents=True, exist_ok=False)
    if role in ['document', 'library']:
        author = json.loads((O/'author_progress.json').read_bytes())
        assert author['closed_execution_verified'] and author['artifact_audit']
    if role == 'library':
        summary = json.loads((O/'fresh_document_summary.json').read_bytes())
        assert summary['status'] == 'STAGE4_TRAIN_ONLY_REFERENCE_FRESH_DOCUMENT_VERIFIED'
        assert summary['actual_exit_code'] == 0 and summary['findings'] == []
        document = json.loads((O/'document_progress.json').read_bytes())
        assert document['closed_execution_verified'] and document['artifact_audit']
        assert author['artifact_audit']['numeric_signature'] == document['artifact_audit']['numeric_signature']
    bundle = {n: dict(base64=base64.b64encode((B/n).read_bytes()).decode(), sha256=sha(B/n)) for n in SOURCES}
    for n in SOURCES: ast.parse((B/n).read_bytes(), feature_version=(3, 8))
    binds = {n: sha(B/n) for n in BOUND}
    code = r'''
import ast,base64,fcntl,hashlib,json,os,pathlib,shutil,subprocess,sys
r=pathlib.Path(ROOT_LITERAL);assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
sys.path.insert(0,str(r/'scripts'));bundle=BUNDLE_LITERAL;binds=BINDS_LITERAL
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for n,item in bundle.items():
 raw=base64.b64decode(item['base64']);assert hashlib.sha256(raw).hexdigest()==item['sha256'];ast.parse(raw,feature_version=(3,8))
 path=r/n
 if path.exists():assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
 else:
  with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
from step4_train_only_protocol import ROOT,owned,read,sha,atomic,select,RESERVE,LOG
from contact_filter_resources import sample_resources
for n,h in binds.items():assert sha(owned(r/n))==h
role=ROLE_LITERAL;run=RUN_LITERAL;parent=owned(r/'step34_numeric_v1');prefix='step4_train_only_'+run
assert not (parent/run).exists() and not (parent/(run+'_supervision')).exists()
assert not any((r/'receipts'/(prefix+x)).exists() for x in ['_driver.py','_driver.json','_request.json','_outer_exit.json'])
clips,sources,names,binding=select(role)
upper=sum((x['control_intervals']+1)*2152+8192 for x in clips)+64*1024**2
resources=sample_resources(r,None,16*1024**3,RESERVE+LOG)
gpu=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
assert gpu>=20*1024**3 and shutil.disk_usage(parent).free>=RESERVE+LOG+upper
spec=read(r/'configs/tracker-env-spec.json');assert sha(spec['robot_urdf'])==spec['robot_urdf_sha256']
s=dict(schema='textop.stage4_train_only_reference_job.v1',run=run,role=role,input_binding=binding,
 executor_sha256=binds['scripts/export_step4_train_only_references.py'],supervisor_sha256=binds['scripts/supervise_step4_train_only_references.py'],
 bound_sources=binds,urdf_sha256=spec['robot_urdf_sha256'],output_root=str(parent),output_upper_bytes=upper,
 physics_steps_authorized=0,optimizer_updates_authorized=0,generated_training_clips=0,original_VAL_motions_in_training=0)
request=r/'receipts'/(prefix+'_request.json');atomic(request,s)
cmd=['taskset','-c','0-3','/usr/bin/python3','-B',str(r/'scripts/supervise_step4_train_only_references.py'),'--request',str(request)]
outer=r/'receipts'/(prefix+'_outer_exit.json');driver=r/'receipts'/(prefix+'_driver.py')
text='import subprocess,pathlib,json,os\np=subprocess.run('+repr(cmd)+')\nwith pathlib.Path('+repr(str(outer))+').open("x") as f:json.dump({"actual_returncode":p.returncode},f);f.flush();os.fsync(f.fileno())\nraise SystemExit(p.returncode)\n'
with driver.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
lock.close()
with (parent/(run+'_outer.log')).open('xb') as f:q=subprocess.Popen(['/usr/bin/python3','-B',str(driver)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
stat=(pathlib.Path('/proc')/str(q.pid)/'stat').read_text().rsplit(')',1)[1].split()
result=dict(status='STAGE4_TRAIN_ONLY_REFERENCE_DISPATCHED_NOT_ACCEPTED',run=run,driver_pid=q.pid,driver_start_ticks=stat[19],
 request_sha256=sha(request),request=s,resources=resources,gpu_free_bytes=gpu,output_free_bytes=shutil.disk_usage(parent).free)
atomic(r/'receipts'/(prefix+'_driver.json'),result);print(json.dumps(result))
'''.replace('ROOT_LITERAL', repr(ROOT)).replace('BUNDLE_LITERAL', repr(bundle)).replace('BINDS_LITERAL', repr(binds)).replace('ROLE_LITERAL', repr(role)).replace('RUN_LITERAL', repr(run_name(role)))
    ast.parse(code, feature_version=(3, 8)); (out/'dispatch_source.py').write_text(code, encoding='utf-8')
    result = json.loads(recorded_remote(code, out/'client.json', timeout=180)); write(out/'dispatch.json', result)
    return result


def collect(role, evidence):
    out = O/evidence; out.mkdir(parents=True, exist_ok=False)
    code = r'''
import array,ast,fcntl,hashlib,json,math,os,pathlib,shutil,struct,sys,time,zipfile
r=pathlib.Path(ROOT_LITERAL);sys.path.insert(0,str(r/'scripts'))
from step4_train_only_protocol import ROOT,owned,read,sha,select
run=RUN_LITERAL;role=ROLE_LITERAL;parent=owned(r/'step34_numeric_v1');prefix='step4_train_only_'+run
request=owned(r/'receipts'/(prefix+'_request.json'));s=read(request);log=parent/(run+'_supervision');files={};hashes={}
for k,p in dict(receipt=parent/run/'receipt.json',exit=log/'exit.json',process=log/'process_exit.json',outer=r/'receipts'/(prefix+'_outer_exit.json')).items():
 if p.exists():files[k]=read(p);hashes[k]=sha(p)
workers=[]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  argv=(proc/'cmdline').read_bytes().decode().split('\0')
  if any(str(r/'scripts'/n) in argv for n in ['export_step4_train_only_references.py','supervise_step4_train_only_references.py']) and str(request) in argv:
   st=(proc/'stat').read_text().rsplit(')',1)[1].split();workers.append(dict(pid=int(proc.name),start_ticks=st[19],state=st[0],argv=argv))
 except (FileNotFoundError,ProcessLookupError):pass
lock=(r/'receipts/full_library_capacity.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
except BlockingIOError:free=False
finally:lock.close()
closed=all(k in files for k in ['receipt','exit','process','outer']) and files['outer']['actual_returncode']==0 and files['process']['child_exit_code']==0 and files['exit']['supervision_passed'] and not workers and free
audit=None
if closed:
 rec=files['receipt'];ex=files['exit'];assert rec['status']=='STAGE4_ORIGINAL_TRAIN_ISAAC_REFERENCES_FINISHED' and rec['output_writers_closed']
 assert hashes['receipt']==ex['receipt_sha256'] and hashes['process']==ex['process_exit_sha256']
 assert sha(log/'resources.jsonl')==ex['resource_sha256'] and sha(request)==ex['request_sha256']
 for k in ['exit','process']:
  x=files[k];assert x['child_reaped'] and x['child_exit_code']==0 and x['sample_errors']==[] and not x['timed_out']
 samples=[json.loads(x) for x in (log/'resources.jsonl').read_text().splitlines()]
 assert samples[-1]['phase']=='after_exit' and samples[-1]['owned_group_processes']==[] and samples[-1]['owned_group_gpu_memory']==[]
 m=rec['private_manifest'];mp=owned(m['path']);assert sha(mp)==m['sha256'] and mp.stat().st_size==m['bytes']
 clips,sources,names,binding=select(role);rows=[json.loads(x) for x in mp.read_text().splitlines()];assert len(rows)==len(clips)
 total=frames=0;signatures=[]
 for entry,row in zip(clips,rows):
  assert all(row[k]==v for k,v in entry.items());p=owned(row['file']);assert p.parent==parent/run
  assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
  with zipfile.ZipFile(p) as z:
   assert z.testzip() is None and set(z.namelist())=={k+'.npy' for k in row['array_sha256']}
   for key,h in row['array_sha256'].items():
    raw=z.read(key+'.npy');assert raw[:6]==b'\x93NUMPY';ver=raw[6:8]
    n=struct.unpack('<H' if ver==b'\x01\x00' else '<I',raw[8:10 if ver==b'\x01\x00' else 12])[0]
    start=10 if ver==b'\x01\x00' else 12;head=ast.literal_eval(raw[start:start+n].decode('latin1'))
    assert list(head['shape'])==row['array_shapes'][key]
    dtype={'float32':'<f4','float64':'<f8','int64':'<i8'}[row['array_dtypes'][key]]
    assert head['descr']==dtype
    payload=raw[start+n:]
    if head['fortran_order']:
     assert len(head['shape'])==2 and dtype=='<f4' and sys.byteorder=='little'
     nr,nc=head['shape'];v=array.array('f');v.frombytes(payload);digest=hashlib.sha256()
     for i in range(nr):digest.update(struct.pack('<'+str(nc)+'f',*[v[j*nr+i] for j in range(nc)]))
     assert digest.hexdigest()==h
    else:assert hashlib.sha256(payload).hexdigest()==h
  signatures.append(dict(array_sha256=row['array_sha256'],array_shapes=row['array_shapes'],array_dtypes=row['array_dtypes']))
  total+=row['bytes'];frames+=row['frames']
 assert total==rec['output_npz_bytes'] and frames==rec['exported_frames']
 signature=hashlib.sha256(json.dumps(signatures,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 audit=dict(numeric_signature=signature,all_npz_bytes_sha_CRC_array_hashes_shapes_dtypes_verified=True,
  clips=len(rows),frames=frames,bytes=total,original_VAL_motions_in_training=0,generated_training_clips=0,model_loads=0,
  producer_finite_checks=True,source_join_exact=True)
tails={}
for p in [log/'child.log',parent/(run+'_outer.log')]:
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-1800));tails[p.name]=f.read().decode(errors='replace')
print(json.dumps(dict(run=run,role=role,observed_unix=time.time(),files=files,sha256=hashes,workers=workers,
 common_lock_free=free,closed_execution_verified=bool(closed),artifact_audit=audit,logs=tails,data_free_bytes=shutil.disk_usage(r).free)))
'''.replace('ROOT_LITERAL', repr(ROOT)).replace('RUN_LITERAL', repr(run_name(role))).replace('ROLE_LITERAL', repr(role))
    ast.parse(code, feature_version=(3, 8))
    result = json.loads(recorded_remote(code, out/'client.json', timeout=900 if role == 'library' else 180))
    write(out/'progress.json', result)
    (O/(role+'_progress.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['refs_launch', 'refs_collect'])
    p.add_argument('--role', choices=['author', 'document', 'library'], required=True); p.add_argument('--evidence', required=True)
    a = p.parse_args(); assert a.evidence.isascii() and a.evidence.isidentifier()
    result = launch(a.role, a.evidence) if a.action == 'refs_launch' else collect(a.role, a.evidence)
    print(json.dumps(result, ensure_ascii=False))

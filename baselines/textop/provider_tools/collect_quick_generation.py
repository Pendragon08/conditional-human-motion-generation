"""Verify/collect closed trained-generation artifacts; keep all arrays private."""
import argparse,hashlib,json,pathlib,shutil,subprocess
from session import remote
B=pathlib.Path(__file__).resolve().parents[1];D=pathlib.Path('D:/具身智能/TextOp复现_20261024');p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();assert a.run.isidentifier() and a.run.isascii()
script=r'''
import pathlib,hashlib,json
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');run=RUN;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();out=r/'generated_reference_private'/run
rp=out/'receipt.json';j=json.loads(rp.read_text());ep=r/'receipts'/('quick_generation_'+run+'_exit.json');ex=json.loads(ep.read_text());assert ex['evaluation_receipt_sha256']==sha(rp)
assert ex['child_exit_code']==0 and ex['supervision_passed'] and ex['child_reaped'] and not ex['timed_out'];assert j['status']=='TRAINED_QUICK_TEXT_REFERENCE_GENERATED' and j['output_writers_closed']
assert sha(r/'receipts'/('quick_generation_'+run+'_process_exit.json'))==ex['process_exit_sha256'] and sha(r/'logs'/('quick_generation_'+run+'_resources.jsonl'))==ex['resources_sha256']
f=pathlib.Path(j['artifact']['file']);assert f.parent==out and f.stat().st_size==j['artifact']['bytes'] and sha(f)==j['artifact']['sha256']
mp=out/'generation.private.json';assert sha(mp)==j['private_metadata_sha256']
print(json.dumps({'receipt':j,'receipt_sha256':sha(rp),'exit':ex,'exit_sha256':sha(ep),'private_files':{str(f):j['artifact']['sha256'],str(mp):sha(mp)}}))
'''.replace('RUN',repr(a.run))
j=json.loads(remote(script,timeout=50));private=D/'datasets/generated_reference_private_20261026'/a.run;private.mkdir(parents=True,exist_ok=True)
for path,h in j['private_files'].items():
 target=private/pathlib.PurePosixPath(path).name
 if not target.exists():
  assert shutil.disk_usage(private).free>1024**3
  z=subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','-i','C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32:'+path,str(target)],capture_output=True,text=True,timeout=90);assert z.returncode==0,z.stderr
 assert hashlib.sha256(target.read_bytes()).hexdigest()==h
import numpy as np
with np.load(private/'generated.private.npz',allow_pickle=False) as z:
 q=np.array(z['qpos_29']);normalized=np.array(z['normalized_features']);features=np.array(z['features']);embedding=np.array(z['text_embeddings'])
assert q.shape==(258,36) and normalized.shape==features.shape==(258,69) and embedding.shape==(2,512)
assert all(np.isfinite(x).all() for x in [q,normalized,features,embedding]) and np.max(np.abs(np.linalg.norm(q[:,3:7],axis=1)-1))<1e-5
meta=json.loads((private/'generation.private.json').read_text(encoding='utf-8'));assert len(meta['joint_names'])==29 and len(set(meta['joint_names']))==29 and len(meta['generation_records'])==32
for i,row in enumerate(meta['generation_records']):assert row['primitive']==i and row['phase']==i//16 and row['normalized_history_sha256']==hashlib.sha256(normalized[i*8:i*8+2].tobytes()).hexdigest()
out=B/'results/quick_generation_20261026'/a.run;out.mkdir(parents=True,exist_ok=True)
for name,value in [('receipt.json',j['receipt']),('exit.json',j['exit']),('collection.json',{'receipt_sha256':j['receipt_sha256'],'exit_sha256':j['exit_sha256'],'private_artifacts_server_and_local_verified':True,'history_fingerprints_verified':32,'private_arrays_published':False})]:
 (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
shutil.copytree(out,D/'results/quick_generation_20261026'/a.run,dirs_exist_ok=True)
print(json.dumps({'run':a.run,'status':j['receipt']['status'],'frames':len(q),'optimizer_updates':0,'physics_steps':0,'private_two_location_sha_verified':True,'root_height_min_m':j['receipt']['root_height_min_m'],'max_joint_difference_speed_rad_s':j['receipt']['max_joint_difference_speed_rad_s'],'mean_primitive_latency_seconds':j['receipt']['primitive_latency_mean_seconds'],'real_time_or_closed_loop_verified':False}))

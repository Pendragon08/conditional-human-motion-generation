"""Bind genuinely completed/privately backed up VAE2000+LDM3000 for generation."""
import ast,base64,hashlib,json,pathlib
from session import remote
B=pathlib.Path(__file__).resolve().parents[1];D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
names=['scripts/generate_quick_reference.py','scripts/supervise_quick_generation.py']
for n in names:ast.parse((B/n).read_text(encoding='utf-8'))
payload={n:base64.b64encode((B/n).read_bytes()).decode() for n in names};backs={}
for stage,target in [('vae',2000),('ldm',3000)]:
 saved=json.loads((B/('results/quick_'+stage+'_20261026')/('quick_'+stage+'_train_v1')/'checkpoint_backup.json').read_text(encoding='utf-8'))['checkpoints']
 ck=[x for x in saved if x['manager_step']==target];assert len(ck)==1 and ck[0]['private_server_and_local_sha_verified'];backs[stage]=ck[0]['sha256']
 f=D/('checkpoints/private/quick_'+stage+'_20261026')/(stage+'_state_'+str(target)+'.pth');assert f.stat().st_size==ck[0]['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==ck[0]['sha256']
script=r'''
import ast,base64,fcntl,hashlib,json,os,pathlib,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert os.getuid()==1013
out=r/'receipts/quick_generation_stage_v1.json';assert not out.exists()
lk=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert shutil.disk_usage(r).free>2*1024**3+64*1024**2
backups=BACKUPS;models={}
for stage,target in [('vae',2000),('ldm',3000)]:
 rp=r/'quick_generator_20261026'/(stage+'_runs')/('quick_'+stage+'_train_v1')/'receipt.json';ep=r/'receipts'/('quick_'+stage+'_quick_'+stage+'_train_v1_exit.json')
 j=json.loads(rp.read_text());ex=json.loads(ep.read_text());assert ex['evaluation_receipt_sha256']==sha(rp) and ex['child_exit_code']==0 and ex['supervision_passed'] and j['output_writers_closed']
 assert j['status']==('QUICK_VAE_2000_TRAINING_FINISHED' if stage=='vae' else 'QUICK_LDM_3000_TRAINING_FINISHED') and j['completed_primitive_attempts']==target
 if stage=='vae':assert j['vae_quality_gate_passed']
 ck=[x for x in j['checkpoints'] if x['manager_step']==target];assert len(ck)==1 and sha(ck[0]['path'])==backups[stage]==ck[0]['sha256']
 models[stage]={'receipt':str(rp),'receipt_sha256':sha(rp),'exit':str(ep),'exit_sha256':sha(ep),'checkpoint':ck[0]}
payload=PAYLOAD
for n,enc in payload.items():
 p=r/n;assert not p.exists();raw=base64.b64decode(enc);ast.parse(raw.decode());p.write_bytes(raw)
corpus=r/'quick_generator_20261026/corpus_v1'
bound=['scripts/paper69_runtime.py','scripts/generator_training_state.py','configs/paper69-env-spec.json','configs/quick-ldm-runtime-spec.json']
spec={'schema':'textop.quick_generated_reference.v1','generation_enabled':True,'sdk_export_enabled':False,'physical_execution_enabled':False,'seed':20261026,'trained_models':models,
 'corpus_dir':str(corpus),'corpus_receipt_sha256':sha(corpus/'corpus_receipt.json'),'bound_sources':{n:sha(r/n) for n in bound},
 'text_schedule':[{'text':'walk forward','primitives':16},{'text':'wave your right hand','primitives':16}],
 'executor_sha256':sha(r/'scripts/generate_quick_reference.py'),'supervisor_sha256':sha(r/'scripts/supervise_quick_generation.py'),
 'scope':'Offline two-text actual trained generation; neutral initial2 then ownreference history,32x8 future frames, no posteriorfuture, no text-semantic/physical/real-time claim'}
sp=r/'configs/quick-generated-reference-spec.json';assert not sp.exists();sp.write_text(json.dumps(spec,indent=2)+'\n')
proof={'status':'QUICK_TRAINED_GENERATION_PROFILE_STAGED','spec_sha256':sha(sp),'spec_base64':base64.b64encode(sp.read_bytes()).decode(),'training_models':models,'new_optimizers':0,'new_sdk_runs':0,'sdk_export_enabled':False,'physical_execution_enabled':False}
out.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''.replace('BACKUPS',repr(backs)).replace('PAYLOAD',repr(payload))
j=json.loads(remote(script,timeout=90));(B/'configs/quick-generated-reference-spec.json').write_bytes(base64.b64decode(j['spec_base64']));out=B/'results/quick_generation_20261026';out.mkdir(exist_ok=True);(out/'stage_v1.json').write_text(json.dumps(j,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in j.items() if k!='spec_base64'}))

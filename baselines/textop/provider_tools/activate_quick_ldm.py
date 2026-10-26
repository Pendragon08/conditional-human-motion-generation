"""Bind both genuine source-matched LDM witnesses, enable only quick training."""
import base64,json,pathlib
from session import remote
B=pathlib.Path(__file__).resolve().parents[1]
script=r'''
import base64,fcntl,hashlib,json,os,pathlib,shutil,subprocess,sys,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');sys.path.insert(0,str(r/'scripts'))
from checkpoint_storage_guard import check
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert os.getuid()==1013
out=r/'receipts/quick_ldm_training_activation_v1.json';assert not out.exists()
assert not(r/'quick_generator_20261026/ldm_runs/quick_ldm_train_v1').exists()
lk=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
sp=r/'configs/quick-ldm-runtime-spec.json';before=sp.read_bytes();spec=json.loads(before);assert not spec['train_authorized'] and not spec['witnesses']
assert sha(r/'scripts/run_quick_ldm.py')==spec['entrypoint_sha256'] and sha(r/'scripts/supervise_quick_ldm.py')==spec['supervisor_sha256']
assert all(sha(r/n)==h for n,h in spec['bound_sources'].items())
plan=json.loads((r/'configs/quick-reproduction-plan-20261026.json').read_text());frozen=dict(plan['frozen_runtime_files'],**plan['control_sources']);assert all(sha(r/n)==h for n,h in frozen.items())
assert sha(spec['vae_checkpoint']['path'])==spec['vae_checkpoint']['sha256'] and sha(pathlib.Path(spec['checkpoint_dir'])/'vae.pth')==spec['vae_checkpoint']['sha256']
assert sha(r/'receipts/quick69_data_acceptance_v1.json')==spec['data_acceptance_sha256']
assert sha(pathlib.Path(spec['corpus_dir'])/'corpus_receipt.json')==spec['corpus_receipt_sha256']
v=spec['vae_training'];assert sha(v['receipt_path'])==v['receipt_sha256'] and sha(v['exit_path'])==v['exit_sha256']
before_sha=sha(sp);witnesses=[]
for name in ['ldm_kernel_author_v2','ldm_kernel_document_v1']:
 rp=r/'quick_generator_20261026/ldm_runs'/name/'receipt.json';ep=r/'receipts'/('quick_ldm_'+name+'_exit.json');j=json.loads(rp.read_text());ex=json.loads(ep.read_text())
 assert ex['evaluation_receipt_sha256']==sha(rp) and ex['child_exit_code']==0 and ex['supervision_passed'] and ex['child_reaped'] and not ex['timed_out'] and ex['supervisor_error'] is None
 assert sha(r/'receipts'/('quick_ldm_'+name+'_process_exit.json'))==ex['process_exit_sha256'] and sha(r/'logs'/('quick_ldm_'+name+'_resources.jsonl'))==ex['resources_sha256']
 assert j['status']=='QUICK_LDM_WITNESS_PASSED' and j['output_writers_closed'] and j['mode']=='witness' and j['optimizer_updates']==4 and j['skipped_updates']==0
 assert j['spec_sha256']==before_sha and j['source_sha256']==spec['entrypoint_sha256'] and j['vae_checkpoint_sha256']==spec['vae_checkpoint']['sha256']
 assert j['corpus_receipt_sha256']==spec['corpus_receipt_sha256'] and j['data_acceptance_sha256']==spec['data_acceptance_sha256']
 m=j['memory_witness'];assert m['model_and_adam_roundtrip_exact'] and m['all_rngs_restored_exactly'] and m['next_batch_replayed_exactly'] and m['pending_evaluation_completed_without_updates'] and m['vae_frozen_and_unchanged']
 assert m['validation_optimizer_updates']==0 and not m['ema_enabled'] and not m['weights_reused_for_training']
 q=j['five_step_cfg_sampling_witness'];assert q['diffusion_steps']==5 and q['cfg']==5 and q['batch']==12 and q['posterior_future_not_used']
 witnesses.append({'run':name,'receipt_sha256':sha(rp),'exit_sha256':sha(ep)})
budget=check(pathlib.Path(spec['checkpoint_dir']),pathlib.Path(spec['checkpoint_allowed_root']),spec['checkpoint_policy'],remaining_saves=4)
assert not list(pathlib.Path(spec['checkpoint_dir']).glob('ldm_state_*.pth')) and not list(pathlib.Path(spec['checkpoint_dir']).glob('emergency_*.pth'))
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True);assert {int(x.split(',')[0]):int(x.split(',')[1]) for x in gpus.splitlines()}[1]>20480
spec['train_authorized']=True;spec['witnesses']=witnesses;after=(json.dumps(spec,indent=2)+'\n').encode();assert {k:v for k,v in json.loads(before).items() if k not in ['train_authorized','witnesses']}=={k:v for k,v in spec.items() if k not in ['train_authorized','witnesses']}
sp.write_bytes(after)
proof={'status':'QUICK_LDM3000_TRAINING_AUTHORIZED_REAL_WITNESSES_BOUND','time_unix':time.time(),'before_spec_sha256':hashlib.sha256(before).hexdigest(),'after_spec_sha256':sha(sp),'before_base64':base64.b64encode(before).decode(),'after_base64':base64.b64encode(after).decode(),'witnesses':witnesses,'storage':budget,'old19_unchanged':all(sha(r/n)==h for n,h in frozen.items()),'training_dispatched':False,'new_optimizer_updates':0,'full_paper_training':False}
out.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''
j=json.loads(remote(script,timeout=90));(B/'configs/quick-ldm-runtime-spec.json').write_bytes(base64.b64decode(j['after_base64']))
out=B/'results/quick_ldm_20261026';out.mkdir(exist_ok=True);(out/'training_activation_v1.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in j.items() if k not in ['before_base64','after_base64']}))

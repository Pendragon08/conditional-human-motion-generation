"""Enable only after two real closed GPU witnesses and actual HOME acceptance."""
import ast
import base64
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B=pathlib.Path(__file__).resolve().parents[1]
O=B/'results/semantic_evaluator_20261029'
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
bindings=[]
for role in ['author','document']:
    run='tmr69d_real_witness_'+role+'_v1'
    j=json.loads((O/run/'progress.json').read_bytes())
    assert j['closed_execution_verified'] and j['artifact_audit']['real_data_fullstate_replay_and_zero_validation_updates_verified']
    if role=='document':assert j['artifact_audit']['cross_process_real_input_model_adamw_rng_loss_exact']
    bindings.append(dict(run=run,sha256=j['sha256']))
sources={}
for name in ['scripts/train_tmr69d_evaluator.py','scripts/supervise_tmr69d_evaluator_training.py']:
    raw=(B/name).read_bytes()
    ast.parse(raw,feature_version=(3,8))
    sources[name]=dict(raw=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
out=O/'tmr_training_activation_v1'
out.mkdir(exist_ok=False)
code=r'''
import os,sys,pathlib,json,hashlib,base64,fcntl,shutil,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));os.sched_setaffinity(0,{0,1,2,3})
from ldm100k_verified_input import owned,read,sha
from contact_filter_resources import sample_resources
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
resources=sample_resources(r,None,16*1024**3,2*1024**3+128*1024**2+16*1024**2)
gpu=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2;assert gpu>=20*1024**3
parent=owned(r/'evaluator_tmr_20261029');bindings=BINDINGS_LITERAL;accepted=[];requests=[]
for b in bindings:
 run=b['run'];log=parent/(run+'_supervision');paths=dict(receipt=parent/run/'receipt.json',exit=log/'exit.json',process=log/'process_exit.json',outer=r/'receipts'/('tmr_real_witness_'+run+'_outer_exit.json'))
 for key,path in paths.items():assert sha(owned(path))==b['sha256'][key]
 rec=read(paths['receipt']);ex=read(paths['exit']);proc=read(paths['process']);outer=read(paths['outer'])
 assert rec['status']=='TMR69D_REAL_DATA_GPU_WITNESS_VERIFIED' and rec['output_writers_closed']
 assert ex['supervision_passed'] and proc['child_reaped'] and proc['child_exit_code']==outer['actual_returncode']==0
 assert ex['receipt_sha256']==b['sha256']['receipt'] and ex['process_exit_sha256']==b['sha256']['process'] and sha(log/'resources.jsonl')==ex['resource_sha256']
 samples=[json.loads(x) for x in (log/'resources.jsonl').read_text().splitlines()];assert samples[-1]['phase']=='after_exit' and not samples[-1]['owned_group_processes']
 sp=r/'receipts'/('tmr_real_witness_'+run+'_request.json');req=read(owned(sp));assert sha(sp)==rec['request_sha256']==ex['request_sha256']
 assert sha(r/'scripts/run_tmr69d_real_witness.py')==rec['source_sha256']==req['executor_sha256'] and sha(r/'scripts/supervise_tmr69d_real_witness.py')==ex['source_sha256']==req['supervisor_sha256']
 for name,value in req['bound_sources'].items():assert sha(owned(r/name))==value
 assert rec['actual_diagnostic_optimizer_calls']==4 and rec['selected_logical_optimizer_steps']==3
 assert rec['actual_formal_evaluator_updates']==rec['generator_or_teacher_updates']==rec['persistent_fullstate_writes']==rec['actual_validation_optimizer_updates']==0
 assert rec['exact_model_adamw_all_rng_sampler_nextbatch_loss_replay'] and rec['source_arrays_norm_and_public_model_fingerprints_unchanged'] and rec['diagnostic_weights_discarded_not_formal_initialization']
 accepted.append(rec);requests.append(req)
for key in ['runtime_binding_sha256','actual_input_signature','actual_input_signature_sha256','actual_initial_model_sha256','actual_final_model_sha256','final_model_adamw_rng_sampler_content_sha256','actual_losses','actual_batch_sha256','internal_validation_loss','internal_validation_batch_sha256']:
 assert accepted[0][key]==accepted[1][key],key
assert not (r/'configs/tmr69d-evaluator-train-spec.json').exists()
home=pathlib.Path('/home/ouyanghao/textop-tmr69d-evaluator-20261029');assert not home.exists()
assert shutil.disk_usage('/home/ouyanghao').free>=2*1024**3+128*1024**2+3*201326592
home.mkdir(mode=0o700);owned(home,pathlib.Path('/home/ouyanghao'));ck=home/'checkpoints';ck.mkdir(mode=0o700)
assert subprocess.check_output(['findmnt','-T',str(home),'-n','-o','FSTYPE'],text=True).strip()=='ext4'
probe=home/'ownership_capacity_probe.bin'
with probe.open('xb') as f:f.write(b'tmr69d_owned_storage_v1');f.flush();os.fsync(f.fileno())
capacity=dict(path=str(ck),uid=ck.stat().st_uid,filesystem='ext4',symlinks=False,writable_fsync_probe_sha256=sha(probe),free_bytes=shutil.disk_usage(home).free,
 required_bytes=2*1024**3+128*1024**2+3*201326592,slots=3,per_slot_upper_bytes=201326592,reserve_bytes=2*1024**3,log_allowance_bytes=128*1024**2)
assert capacity['free_bytes']>=capacity['required_bytes'] and capacity['uid']==1013
sources=SOURCES_LITERAL
for name,item in sources.items():
 path=r/name;assert not path.exists();raw=base64.b64decode(item['raw']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
spec=dict(schema='textop.train_only_robot_tmr_evaluator.v1',run='tmr69d_evaluator500_train_v1',formal_training_authorized=True,
 inputs=requests[0]['inputs'],bound_sources=requests[0]['bound_sources'],upstream_model_dir=requests[0]['upstream_model_dir'],upstream_model_sha256=requests[0]['upstream_model_sha256'],
 witness_bindings=bindings,accepted_input_signature_sha256=accepted[0]['actual_input_signature_sha256'],diagnostic_initial_model_sha256=accepted[0]['actual_initial_model_sha256'],
 seed=20261102,epochs=500,batch_size=32,gradient_samples=32462,internal_validation_samples=1606,expected_optimizer_updates=507500,
 optimizer=dict(type='torch.optim.AdamW',lr=.0001,weight_decay=.01),lr_schedule='constant upstream configure_optimizers',
 checkpoint_dir=str(ck),checkpoint_upper_bytes=201326592,save_epochs=[100,500],immutable_emergency_slots=1,
 reserve_bytes=2*1024**3,home_log_allowance_bytes=128*1024**2,storage_acceptance=capacity,
 early_stopping=False,original_validation_used_for_gradient=False,formal_weights_seeded_from_diagnostic=False,
 executor_sha256=sources['scripts/train_tmr69d_evaluator.py']['sha256'],supervisor_sha256=sources['scripts/supervise_tmr69d_evaluator_training.py']['sha256'])
policy={k:spec[k] for k in ['inputs','bound_sources','upstream_model_sha256','seed','epochs','batch_size','expected_optimizer_updates','save_epochs','checkpoint_upper_bytes']}
spec['runtime_binding_sha256']=hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':')).encode()).hexdigest()
raw=(json.dumps(spec,indent=2,allow_nan=False)+'\n').encode();sp=r/'configs/tmr69d-evaluator-train-spec.json'
with sp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
proof=dict(status='TMR69D_FORMAL_TRAINING_ENABLED_FROM_TWO_REAL_WITNESSES',spec_sha256=sha(sp),storage_acceptance=capacity,resources=resources,gpu1_free_bytes=gpu,training_dispatched=False,
 actual_formal_evaluator_updates=0,original_generator_teacher_training_updates=0)
with (parent/'training_activation_v1.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(proof,spec_raw_base64=base64.b64encode(raw).decode())))
'''.replace('BINDINGS_LITERAL',repr(bindings)).replace('SOURCES_LITERAL',repr(sources))
ast.parse(code,feature_version=(3,8))
j=json.loads(recorded_remote(code,out/'activation_client.json',timeout=180))
raw=base64.b64decode(j.pop('spec_raw_base64'))
assert hashlib.sha256(raw).hexdigest()==j['spec_sha256']
spec=B/'configs/tmr69d-evaluator-train-spec.json'
assert not spec.exists()
spec.write_bytes(raw)
(out/'activation.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
print(json.dumps(j))

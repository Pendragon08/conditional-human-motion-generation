"""Prepare a DISABLED small-generator proposal from pinned settings and storage.

No models/checkpoints are loaded, no server paths/configs are written, and no
corpus is accepted. Actual5000 teacher/pilot acceptance still precedes training.
"""
import argparse, copy, datetime, hashlib, json, pathlib, re
from session import remote
B=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
assert re.fullmatch('[a-z0-9_]+',a.run)
R=B/'results/quick_generator_plan_20261026'/a.run;R.mkdir(parents=True,exist_ok=False)
proposal=B/'configs/quick-generator-proposal-20261026.json';assert not proposal.exists()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
snapshot=json.loads(remote('''import hashlib,json,os,pathlib,shutil,subprocess,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=json.loads((r/'configs/quick-reproduction-plan-20261026.json').read_text())
assert all(sha(r/f)==v for f,v in q['frozen_runtime_files'].items()) and all(sha(r/f)==v for f,v in q['control_sources'].items())
s=json.loads((r/'configs/paper-step2.json').read_text());paths={}
for name,p in [('data',r),('home',pathlib.Path('/home/ouyanghao'))]:
 paths[name]=dict(path=str(p),resolved=str(p.resolve()),uid=p.stat().st_uid,device=p.stat().st_dev,
 writable=os.access(p,os.W_OK),free_bytes=shutil.disk_usage(p).free,
 filesystem_type=subprocess.check_output(['findmnt','-no','FSTYPE','-T',str(p)],text=True).strip())
progress=json.loads((r/'tracker_validation/filter_policy_paper69_solid_floor_20261026_v1/receipt.json').read_text())
print(json.dumps(dict(sample_unix=time.time(),file_systems=paths,paper_step2_sha256=sha(r/'configs/paper-step2.json'),
 paper={k:s[k] for k in ['representation','vae','ldm']},base_environment_sha256=sha(r/'configs/paper69-env-spec.json'),
 runtime_sha256={f:sha(r/'scripts'/f) for f in ['paper69_runtime.py','paper69_data_runtime.py','generator_training_state.py']},
 original16_and_controls_unchanged=True,teacher_status=progress['status'],teacher_completed=progress['completed_learning_iterations'],teacher_updates=progress['optimizer_updates'],
 sdk_runs=0,new_optimizer_updates=0,server_writes=0)))
''',timeout=45))
paper=json.loads((B/'configs/paper-step2.json').read_text(encoding='utf-8'))
assert all(snapshot['paper'][k]==paper[k] for k in ['representation','vae','ldm'])
assert snapshot['paper_step2_sha256']=='d668360a8cbc1d4ad48845467841f3d4d9804f291a86790201ac3de74fec42dd'
files=snapshot['file_systems'];assert files['data']['device']!=files['home']['device']
assert all(x['uid']==1013 and x['writable'] and x['path']==x['resolved'] and x['filesystem_type']=='ext4' for x in files.values())
size_path=B/'results/training_storage_20261025/ema_off_audit_20261026/receipt.json'
size=json.loads(size_path.read_text(encoding='utf-8'));assert size['new_checkpoint_files_written']==0 and size['status']=='CHECKPOINT_SIZE_PLANNING_ONLY'
reference=json.loads((B/'results/quick_pilot_candidate_20261026/selection_metadata.json').read_text(encoding='utf-8'))
GIB=1024**3;MIB=1024**2
stages={}
for name,total,parts,ceiling in [('vae',2000,[1000,500,500],600*MIB),('ldm',3000,[1000,1000,1000],300*MIB)]:
 cfg=copy.deepcopy(snapshot['paper'][name]);original_total=cfg['total_manager_steps']
 assert total*100==original_total and sum(parts)==total and all(x%4==0 for x in parts)
 assert [x//100 for x in cfg['stages_from_released_USAGE']]==parts
 cfg.update(total_manager_steps=total,stages=parts,paper_total_manager_steps=original_total,
            paper_stages=cfg.pop('stages_from_released_USAGE'),optimizer='Adam' if name=='vae' else 'AdamW',
            ema_enabled=False,anneal_lr=True,save_every=1000,eval_every=500,eval_steps=12,
            evaluation_choice='Quick-only12=three complete4-primitive validation draws; differs released10',
            checkpoint_completed_steps=list(range(1000,total+1,1000)),emergency_checkpoint_slots=1,
            checkpoint_per_file_planning_ceiling_bytes=ceiling,expected_batch_draws=total//4,
            actual_optimizer_updates=None,skipped_updates=None,completed_primitive_attempts=0,
            checkpoint_size_previous_ema_off_proxy_bytes=size['stages'][name]['per_checkpoint_planning_bytes_with_one_percent_margin'])
 cfg['total_checkpoint_planning_bytes']=(len(cfg['checkpoint_completed_steps'])+1)*ceiling
 stages[name]=cfg
budgets={
 'data':dict(checkpoints=stages['vae']['total_checkpoint_planning_bytes'],future_teacher5000_full_and_inference=192*MIB,
             accepted_quick_corpus_and_clip=GIB,private_trace_and_video_allowance=512*MIB,logs_metadata=512*MIB,free_reserve=2*GIB),
 'home':dict(checkpoints=stages['ldm']['total_checkpoint_planning_bytes'],logs_metadata=512*MIB,free_reserve=2*GIB)}
for name,value in budgets.items():
 need=sum(value.values());value.update(planned_total_bytes=need,snapshot_free_bytes=files[name]['free_bytes'],
                                      snapshot_margin_bytes=files[name]['free_bytes']-need)
 assert value['snapshot_margin_bytes']>0
plan=dict(schema='textop.quick_generator_proposal.v1',status='PROPOSED_ONLY_NOT_TRAINING_READY',
 quality_label='QUICK_REPRODUCTION_NOT_PAPER_COMPLETE',launch_enabled=False,corpus_build_enabled=False,
 trainer_entrypoint_for_this_proposal=None,trainer_and_dataset_runtime_validated=False,
 reason='Independent1-percent generator chain proposal for the user-requested quick reproduction, not activated by this planning step',
 paper_recipe_unchanged=True,paper_recipe_server_sha256=snapshot['paper_step2_sha256'],
 teacher_required_completed_iterations=5000,teacher_backup_and_controlled_stop_required=True,
 physical_pilot_required='Real same-contact quick SDK author/fresh smoke and representative pilot; closed exit/resources/whole-trial evidence',
 data_scope=dict(proposed_reference_sources=64,proposed_train_sources_before_physics=47,proposed_val_sources_before_physics=17,
  actual_retained_sources=None,actual_retained_text_pairs=None,corpus_accepted=False,
  representation=snapshot['paper']['representation'],mirror_stays_with_original_source_split=True,
  private_assets_only=True,training_inputs='Retargeted reference motions that pass the scoped quick tracking assessment, with original verified BABEL timing; executed poses are evaluation/video data, not substituted training targets',
  required_data_acceptance=['Exact retained-source/variant/text-time coverage and private persistent copies',
   'Source-family and exact-pose split checks; keep original assets if a view quarantines an overlapping training family',
   'At least35 pose frames per primitive window, no cross-source transitions',
   'Refit69D feature statistics on retained quick TRAIN only; no2296-source unfiltered diagnostic statistics',
   'Real frozenCLIP cache covering all retained nonempty captions; mirrored captions preserved',
   'Record action-weight statistics scope; never silently label a mismatched full-corpus table as quick-train statistics',
   'Validate actual batch512/four primitives/history2/future8 loader, gradients and resumable state on this corpus before dispatch']),
 vae=stages['vae'],ldm=stages['ldm'],
 storage=dict(status='PROVISIONAL_NOT_CREATED_OR_ACTIVATED',
   vae_checkpoint_root='/data/code/ouyanghao/oyh/textop-reproduction-20261024/quick_generator_20261026/vae/checkpoints',
   ldm_checkpoint_root='/home/ouyanghao/textop-quick69-checkpoints-20261026/ldm',volumes=budgets,
   refresh_before_launch_and_each_save=True,no_old_checkpoint_relocation_or_deletion=True,
   per_file_ceiling_is_planning_allowance_not_measurement=True,
   excluded=['Paper final431D actor WBC','Full-corpus generated augmentation','Additional resumes/seeds beyond declared emergency slots','Paper-scale200k/300k checkpoint plan']),
 resources=dict(gpu=1,cpu_affinity=[0,1,2,3],torch_memory_fraction=.65,minimum_gpu_free_gib=20,
                no_new_job_while_existing_teacher_or_pilot_owns_SDK=True,shared_dependencies_unchanged=True),
 activation_requirements=['Genuine5000 handoff and actual pilot results, never CPU/synthetic evidence alone',
  'Implement and validate a separate quick launcher without weakening formal40k/full-generator gates',
  'Bind exact source/corpus/statistics/CLIP/selected trainedVAE/quick-plan hashes in checkpoints',
  'Measure actual loss/rollout stability before progressingVAE toLDM; finite loss alone is not quality acceptance',
  'Record primitive attempts, real optimizer updates and skipped gradients separately; validation updates0',
  'Owner/persistent/writable/non-RAM checks and actual file-size/free-capacity guards, including emergency saves'],
 schedule_difference='Curriculum and LR clocks compressed100x, not just early stopping the paper recipe. Quick weights are not an exact continuation of paper training; any later transfer is a separately labeled warm-start run.',
 first_chain_scope='Text-to-reference then simulated tracking by5k privileged teacher only; not the paper final deployable WBC or complete real-time TextOp system',
 actual_generator_training_started=False,generation_execution_loop_completed=False,actual_result_videos=0,
 source_bindings=dict(runtime_server_sha256=snapshot['runtime_sha256'],base_environment_server_sha256=snapshot['base_environment_sha256'],
  prior_checkpoint_size_audit_sha256=sha(size_path),official_manager_local_sha256=sha(B/'upstream/TextOpRobotMDAR/robotmdar/train/manager.py'),
  official_train_vae_local_sha256=sha(B/'upstream/TextOpRobotMDAR/robotmdar/train/train_mvae.py'),official_train_ldm_local_sha256=sha(B/'upstream/TextOpRobotMDAR/robotmdar/train/train_dar.py')))
proposal.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
receipt=dict(status='QUICK_GENERATOR_DISABLED_PROPOSAL_AND_STORAGE_PREPARED',sample=snapshot,
 proposal_sha256=sha(proposal),paper_numeric_parameters_match=True,paper_full_recipe_modified=False,
 gpu_or_model_work=0,checkpoints_loaded_or_written=0,server_paths_created=0,server_configs_modified=0,
 actual_retained_sources=None,actual_generator_training_started=False,storage=budgets)
(R/'assessment.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=receipt['status'],teacher_completed=snapshot['teacher_completed'],
 planned_manager_attempts={k:v['total_manager_steps'] for k,v in stages.items()},
 volume_margins_bytes={k:v['snapshot_margin_bytes'] for k,v in budgets.items()},launch_enabled=False)))

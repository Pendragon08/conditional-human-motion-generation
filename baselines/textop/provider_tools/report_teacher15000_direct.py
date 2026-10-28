"""Aggregate already-collected evidence; do not operate training or physics."""
import pathlib,json,hashlib,datetime,html,re,shutil
B=pathlib.Path(__file__).resolve().parents[1];D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
O=B/'results/teacher15000_direct_strict_20261027';run='teacher15000_direct_full_v1'
j=json.loads((O/run/'progress.json').read_text(encoding='utf-8'));r=j['files'].get('receipt',{})
stop=json.loads((O/'control/user_stop_v1_result.json').read_text(encoding='utf-8'))
assert stop['common_sdk_lock_released'] and stop['recoverable_completed']==15000
assert r['stage']=='full' and r['policy_checkpoint_completed_iterations']==15000 and r['optimizer_updates']==0
done=r.get('finished_trials',0);passed=r.get('complete_trials',0);failed=r.get('failed_trials',0)
assert done==passed+failed and 0<=done<=8278
pending=8278-done
audit_path=B/'results/filtered15000_handoff_20261027/full_result_audit_v1/progress.json'
audit_verified=False;audit_sha=None
if audit_path.exists():
 aj=json.loads(audit_path.read_text(encoding='utf-8'));ar=aj.get('files',{}).get('receipt',{})
 if aj.get('closed_execution_verified'):
  assert ar['strict_result_accepted'] and ar['source_count']==ar['attempted']==8278 and ar['unattempted']==0
  assert ar['complete_passes']==passed and ar['failed']==failed and done==8278
  assert ar['raw_execution_receipt_sha256']==j['sha256']['receipt'] and j['full_execution_closed']
  audit_verified=True;audit_sha=aj['sha256']['receipt']
handoff_stage='WAITING_FULL_REPLAY' if not audit_verified else 'FILTERED_CORPUS_ACCEPTANCE_PENDING'
generator_started=False;corpus_accepted=False;generator_attempts=0;handoff_workers=[]
handoff_lock_free=j['common_lock_free'];handoff_observed=j['observed_unix'];generator_spec_sha=None
if audit_verified:
 for next_run in ['filtered15000_corpus_author_v1','filtered15000_corpus_document_v1','filtered15000_recovery_v1','filtered15000_acceptance_v1','filtered15000_vae_author_v1','filtered15000_vae_document_v1','filtered15000_vae200k_train_v1']:
  cp=B/'results/filtered15000_handoff_20261027'/next_run/'progress.json'
  if cp.exists():
   cj=json.loads(cp.read_text(encoding='utf-8'));cr=cj.get('files',{}).get('receipt',{})
   handoff_stage=cr.get('status','POST_FILTER_JOB_DISPATCHED_NOT_YET_VERIFIED');handoff_workers=cj.get('workers',[])
   handoff_lock_free=cj.get('common_lock_free',False);handoff_observed=max(handoff_observed,cj['observed_unix'])
   if next_run=='filtered15000_vae200k_train_v1':
    generator_attempts=cr.get('completed_primitive_attempts',0);generator_spec_sha=cr.get('spec_sha256')
    generator_started=generator_attempts>0 and cr.get('status') in ['RUNNING_FILTERED15000_VAE_TRAINING','FILTERED15000_VAE_200000_TRAINING_FINISHED']
 readiness_path=B/'results/filtered15000_handoff_tools_20261027/vae_train_activation.json'
 if readiness_path.exists():
  ready=json.loads(readiness_path.read_text(encoding='utf-8')).get('readiness',{})
  corpus_accepted=ready.get('filtered_corpus_accepted') is True and ready.get('formal_training_ready') is True
user_100k_target=False;user_control_status=None;user_ldm_control_status=None;vae_selected_step=None;ldm_attempts=0;ldm_target=300000
policy_path=B/'configs/teacher15000-full-to-generator-autostart-20261027.json'
if policy_path.exists():
 pol=json.loads(policy_path.read_bytes());user_100k_target=pol.get('training',{}).get('fresh_filtered_vae_primitive_steps')==100000;ldm_target=pol.get('training',{}).get('then_fresh_filtered_ldm_primitive_steps',300000)
if user_100k_target:
 cc=B/'results/vae100k_user_handoff_20261028/user_vae100k_control_v1/progress.json'
 if cc.exists():
  cj=json.loads(cc.read_bytes());cr=cj.get('files',{}).get('receipt',{});user_control_status=cr.get('status')
  handoff_stage=user_control_status or 'USER_VAE100K_CONTROL_DISPATCHED';handoff_workers=cj.get('workers',[]);handoff_observed=max(handoff_observed,cj['observed_unix']);handoff_lock_free=cj['common_lock_free']
  generator_attempts=cr.get('last_observed_primitive_attempts',generator_attempts)
  if cj.get('closed_execution_verified'):
   vae_selected_step=cr['selected_checkpoint_attempts'];generator_attempts=vae_selected_step
  for kind,stage_run in [('resume','r100_author_v1'),('resume','r100_document_v1'),('resume','user_vae100k_resume_train_v1'),('quality','user_vae100k_quality_v1'),('vae_export','user_vae100k_export_v1'),('ldm','user_vae100k_ldm_author_v1'),('ldm','user_vae100k_ldm_document_v1'),('ldm','user_vae100k_ldm300k_train_v1')]:
   path=B/('results/vae100k_user_handoff_20261028' if kind in ['quality','resume'] else 'results/filtered15000_handoff_20261027')/stage_run/'progress.json'
   if path.exists():
    jn=json.loads(path.read_bytes());rn=jn.get('files',{}).get('receipt',{})
    handoff_stage=rn.get('status','USER_VAE100K_SUCCESSOR_DISPATCHED')
    if stage_run=='user_vae100k_ldm_author_v1' and handoff_stage=='FILTERED15000_LDM_WITNESS_PASSED':handoff_stage='FULL_LDM_AUTHOR_PASSED_FRESH_PENDING'
    if stage_run=='user_vae100k_ldm_document_v1' and handoff_stage=='FILTERED15000_LDM_WITNESS_PASSED':handoff_stage='FULL_LDM_TWO_WITNESSES_PASSED_FORMAL_PENDING'
    generator_spec_sha=rn.get('spec_sha256',generator_spec_sha);handoff_workers=jn['workers'];handoff_lock_free=jn['common_lock_free'];handoff_observed=max(handoff_observed,jn['observed_unix'])
    if stage_run=='user_vae100k_resume_train_v1':
     generator_attempts=rn.get('completed_primitive_attempts',99700);generator_spec_sha=rn.get('spec_sha256')
     if jn.get('closed_execution_verified'):vae_selected_step=rn['selected_checkpoint']['manager_step']
    if stage_run=='user_vae100k_ldm300k_train_v1':ldm_attempts=rn.get('completed_primitive_attempts',0);generator_spec_sha=rn.get('spec_sha256')
 generator_started=True
ldm_successor_run=None;ldm_selected_checkpoint_sha256=None;ldm_dispatch_unconfirmed=False;ldm_latest_confirmed_stage=None
endpoint_path=B/'results/ldm100k_user_handoff_20261028/user_ldm100k_control_v1/progress.json'
if ldm_target==100000 and endpoint_path.exists():
 endpoint=json.loads(endpoint_path.read_bytes());en=endpoint.get('files',{}).get('receipt',{});user_ldm_control_status=en.get('status')
 if user_ldm_control_status=='BOUND_SIGINT_SENT_WAITING_ORIGINAL_EXITS':handoff_stage='LDM100K_SAVED_WAITING_ORIGINAL_EXIT'
 if endpoint.get('closed_execution_verified'):
  assert user_ldm_control_status=='USER_LDM100000_STOP_ACCEPTED_ORIGINAL_EXITS_RETAINED' and en['selected_checkpoint_attempts']==100000
  handoff_stage=user_ldm_control_status;ldm_attempts=100000;handoff_workers=[];handoff_lock_free=endpoint['common_lock_free'];handoff_observed=max(handoff_observed,endpoint['observed_unix'])
  ldm_selected_checkpoint_sha256=en['selected_checkpoint']['sha256']
  ldm_latest_confirmed_stage=handoff_stage
  for successor_run in ['user_ldm100k_prefix_export_v1','user_ldm100k_prefix_document_v1','user_ldm100k_heldout_scope_v1','user_ldm100k_generation_author_v1','user_ldm100k_generation_author_v2','user_ldm100k_generation_author_v3','user_ldm100k_generation_document_v3','user_ldm100k_diagnostic32_v1','user_ldm100k_diagnostic_summary_v2','user_ldm100k_generated_fk_v1','user_ldm100k_generated_fk_v2','user_ldm100k_generated_physics_v1','user_ldm100k_execution_videos_v1','user_ldm100k_tracking_fidelity_v1']:
   successor_path=B/'results/ldm100k_user_handoff_20261028'/successor_run/'progress.json'
   successor=json.loads(successor_path.read_bytes()) if successor_path.exists() else {}
   successor_receipt=successor.get('files',{}).get('receipt',{})
   if successor_receipt.get('status') and successor['observed_unix']>=handoff_observed:
    handoff_stage=successor_receipt['status'];ldm_successor_run=successor_run
    handoff_workers=successor['workers'];handoff_lock_free=successor['common_lock_free'];handoff_observed=successor['observed_unix']
    ldm_latest_confirmed_stage=handoff_stage;ldm_dispatch_unconfirmed=False
   dispatch=successor_path.with_name('dispatch.json')
   if not successor_receipt and dispatch.exists() and not successor_path.with_name('launch.json').exists():
    client=json.loads(dispatch.read_bytes())
    if client.get('client_reaped') and client.get('actual_client_exit_code') not in [None,0]:
     handoff_stage='LDM100K_SUCCESSOR_DISPATCH_UNCONFIRMED';ldm_successor_run=successor_run;ldm_dispatch_unconfirmed=True
     handoff_workers=[];handoff_lock_free=None;handoff_observed=max(handoff_observed,dispatch.stat().st_mtime)
for semantic_run in ['tmr_isolated_runtime_v1','tmr_text_models_v1','tmr69d_training_scope_v1','tmr69d_training_scope_v2','tmr69d_numeric_data_v1','tmr69d_core_contract_v1','tmr_text_models_import_verify_v1','tmr69d_real_witness_author_v1','tmr69d_real_witness_document_v1','tmr69d_evaluator500_train_v1']:
 semantic_path=B/'results/semantic_evaluator_20261029'/semantic_run/'progress.json'
 if semantic_path.exists():
  sj=json.loads(semantic_path.read_bytes());sr=sj.get('files',{}).get('receipt',{})
  if sr.get('status') and sj['observed_unix']>=handoff_observed:
   handoff_stage=sr['status'];ldm_successor_run=semantic_run;handoff_workers=sj['workers'];handoff_lock_free=sj['common_lock_free'];handoff_observed=sj['observed_unix'];ldm_latest_confirmed_stage=handoff_stage;ldm_dispatch_unconfirmed=False
last_verified_handoff_stage=ldm_latest_confirmed_stage if ldm_dispatch_unconfirmed else handoff_stage;access_unavailable=False;access_attempt=None
access_path=B/'results/filtered15000_handoff_20261027/filtered15000_vae200k_train_v1/access_status.json'
if access_path.exists():
 access=json.loads(access_path.read_text(encoding='utf-8'))
 access_unavailable=access.get('status')=='REMOTE_SSH_UNAVAILABLE_TRAINING_CURRENT_STATE_UNKNOWN' and access['attempt_unix']>handoff_observed
 if access_unavailable:
  access_attempt=datetime.datetime.fromtimestamp(access['attempt_unix'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
  handoff_stage='REMOTE_SSH_UNAVAILABLE_CURRENT_TRAINING_UNKNOWN'
access_note='最近远程查询连接超时；下列训练步数为最后已核验值，不能当作当前实时状态。未观察到服务器训练退出，未停止或重启训练。' if access_unavailable else '训练状态以此采样时刻的真实收据为准。'
phase_label='远程暂不可达，当前训练状态未确认' if access_unavailable else '新筛后VAE正式训练中' if handoff_stage=='RUNNING_FILTERED15000_VAE_TRAINING' else handoff_stage
if user_100k_target:
 phase_label={'LDM100K_SAVED_WAITING_ORIGINAL_EXIT':'LDM十万步已完整保存，正在核验退出','USER_LDM100000_STOP_ACCEPTED_ORIGINAL_EXITS_RETAINED':'LDM十万步按用户目标完成，接续效果验收','FULL_LDM_AUTHOR_PASSED_FRESH_PENDING':'LDM首轮启动检查通过，正在独立复查','FULL_LDM_TWO_WITNESSES_PASSED_FORMAL_PENDING':'LDM独立启动检查通过，准备正式开训','STARTING_USER_VAE100K_QUALITY':'VAE十万步质量和恢复检查中','STARTING_FILTERED15000_VAE_CPU_EXPORT':'VAE十万步推理导出检查中','STARTING_FILTERED15000_LDM':'LDM启动检查中','FAILED_VAE100K_CONTROL_RETAIN_ALL_EVIDENCE':'原VAE在99,700步存储保护退出，已保存紧急检查点','STARTING_USER_VAE100K_RESUME':'99,700步检查点恢复与验收中','USER_VAE100K_RESUME_WITNESS_PASSED':'99,700步恢复验收通过，接续补齐300步','RUNNING_USER_VAE100K_RESUME':'从99,700步补齐至100,000步训练中','USER_VAE100000_RESUME_FINISHED':'10万步已完整保存，接续质量验收与LDM','WAITING_FOR_REAL_VAE100000_SAVED_AND_RELOADED':'按用户新目标训练至10万步，等待完整检查点','BOUND_SIGINT_SENT_WAITING_ORIGINAL_EXITS':'10万步已保存，等待原VAE退出核验','USER_VAE100000_STOP_ACCEPTED_ORIGINAL_EXITS_RETAINED':'用户指定10万步已选定，接续质量验收与LDM','USER_VAE100K_QUALITY_VALIDATION_PASSED':'10万步质量验收通过，接续CPU导出','USER_VAE100000_CPU_EXPORT_PASSED':'10万步推理导出完成，接续完整LDM验收','RUNNING_FILTERED15000_LDM_WITNESS':'完整LDM实际验收中','FILTERED15000_LDM_WITNESS_PASSED':'完整LDM验收已完成，接续正式训练','RUNNING_FILTERED15000_LDM_TRAINING':f'LDM正式训练中（目标{ldm_target:,}步）'}.get(handoff_stage,handoff_stage)
phase_label={'STARTING_LDM100K_PREFIX_CPU_EXPORT':'LDM十万步已保存，CPU完整性与推理导出检查中','USER_LDM100000_PREFIX_CPU_EXPORT_PASSED':'LDM十万步CPU导出通过，接续独立复核','FAILED_LDM100K_PREFIX_CPU_EXPORT':'LDM十万步CPU导出失败，保留现场并处理','STARTING_LDM100K_FRESH_CPU_VERIFICATION':'LDM十万步独立CPU复核中','USER_LDM100K_PREFIX_FRESH_CPU_VERIFIED':'LDM十万步独立CPU复核通过，接续留出评估','FAILED_LDM100K_PREFIX_FRESH_CPU_VERIFICATION':'LDM十万步独立CPU复核失败，保留现场并处理','STARTING_LDM100K_HELDOUT_SCOPE':'留出评估范围与来源核验中','USER_LDM100K_HELDOUT_METADATA_SCOPE_BUILT':'留出评估范围核验完成，接续真实生成检查','FAILED_LDM100K_HELDOUT_SCOPE':'留出评估范围构建失败，保留现场并处理','STARTING_LDM100K_HELDOUT_GENERATION':'十万步模型的配对生成效果检查中','USER_LDM100K_GENERATION_WITNESS_FINISHED':'配对生成检查已执行，接续独立复核与诊断','USER_LDM100K_DIAGNOSTIC32_GENERATION_FINISHED':'32组留出动作配对诊断已结束，质量结论仍待汇总','FAILED_LDM100K_HELDOUT_GENERATION':'配对生成检查失败，保留全部输出并处理'}.get(handoff_stage,phase_label)
closure_note='全库结果已由独立有界CPU作业核验，原回放与核验作业均真实退出；生成器后续状态见下方真实收据阶段。' if audit_verified else '进行中统计不是最终筛选结论。未测试动作没有被过滤。'
if ldm_dispatch_unconfirmed:phase_label='LDM十万步已完成，后续评估派发结果待联网核实'
phase_label={'USER_LDM100K_DIAGNOSTIC32_CPU_SUMMARY_FINISHED':'全部配对结果已独立汇总，接续真实物理回放',
 'STARTING_USER_LDM100K_GENERATED_FK':'生成轨迹的机器人正运动学转换中',
 'USER_LDM100K_GENERATED_ISAAC_FK_FINISHED':'全部生成轨迹正运动学转换完成，接续严格物理回放',
 'USER_LDM100K_GENERATED_FK_FAILED_PARTIAL_RETAINED':'生成轨迹转换失败，保留现场并处理',
 'RUNNING_LDM100K_RECORDED_POSE_VIDEOS':'真实执行姿态影片制作中（CPU/Mesa，0新动力学）',
 'USER_LDM100K_RECORDED_EXECUTION_VIDEOS_FINISHED':'真实执行影片已制作，接续完整语义/WBC/在线验收',
 'USER_LDM100K_GENERATED_STRICT_PHYSICS_FINISHED':'全部生成动作严格物理回放已结束，接续视频与完整效果验收'}.get(handoff_stage,phase_label)
if ldm_successor_run=='user_ldm100k_generated_physics_v1' and handoff_stage=='RUNNING_TEACHER15000_STRICT':phase_label='十万步模型生成动作的严格物理回放中（冻结15k策略，0训练）'
phase_label={'USER_LDM100K_RECORDED_TRACKING_FIDELITY_FINISHED':'真实执行姿态跟踪误差核验完成，接续语义评价',
 'TMR_ISOLATED_RUNTIME_BOOTSTRAP_FINISHED':'机器人语义评价器的独立CPU环境已检查，尚未训练评价器',
 'STARTING_PINNED_TMR_TEXT_MODEL_PREPARATION':'正在准备语义评价所需的固定版本文本模型',
 'PINNED_TMR_TEXT_MODELS_CPU_VERIFIED':'文本模型冻结CPU检查通过，接续真实TRAIN语料与评价器验收',
 'FAILED_TMR_TEXT_MODEL_PREPARATION_ALL_PARTIALS_RETAINED':'服务器文本模型直连失败，保留现场并继续本地缓存核验；生成器已完成十万步',
 'TMR69D_TRAINING_METADATA_SCOPE_BUILT':'评价器的TRAIN范围已核验，接续真实数值输入与文本模型检查',
 'TMR69D_NUMERIC_TRAIN_INPUTS_ACCEPTED':'评价器全部原TRAIN数值输入已实际核验，等待文本模型与真实梯度检查',
 'TMR69D_SYNTHETIC_CPU_CORE_CONTRACT_VERIFIED':'原TMR方法的CPU恢复契约通过，继续文本模型上传与真实数据梯度验收',
 'STARTING_IMPORTED_TMR_TEXT_MODELS_CPU_VERIFICATION':'已导入文本模型的冻结CPU检查中',
 'IMPORTED_TMR_TEXT_MODELS_CPU_VERIFIED':'文本模型已冻结CPU核验，接续真实数据梯度与独立恢复验收',
 'TMR69D_REAL_DATA_GPU_WITNESS_VERIFIED':'评价器真实动作/文本梯度与完整恢复检查通过，接续独立复核和正式评价器',
 'PREPARING_FRESH_TMR69D_FORMAL_EVALUATOR':'正在准备独立语义评价器；生成器保持十万步',
 'RUNNING_FRESH_TMR69D_EVALUATOR_TRAINING':'独立语义评价器训练中；VAE与LDM均保持十万步',
 'TMR69D_FORMAL_EVALUATOR_500_EPOCHS_FINISHED':'语义评价器训练已结束，接续冻结推理与真正留出指标',
 'FAILED_TMR69D_FORMAL_EVALUATOR_ALL_STATES_RETAINED':'语义评价器训练退出，保留全部状态并处理；生成器未追加训练'}.get(handoff_stage,phase_label)
diagnostic_summary=None;generated_physics=None;new_effect_note=''
sp=B/'results/ldm100k_user_handoff_20261028/user_ldm100k_diagnostic_summary_v2/progress.json'
if sp.exists():
 sj=json.loads(sp.read_bytes())
 if sj.get('closed_execution_verified'):
  sr=sj['files']['receipt'];diagnostic_summary=dict(statistics=sr['statistics'],actual_pairs=sr['actual_pairs'],actual_trials=sr['actual_trials'],receipt_sha256=sj['sha256']['receipt'],semantic_metrics_computed=False)
  c=sr['statistics']['normalized_huber']['conditional_cfg5']['trial_mean'];u=sr['statistics']['normalized_huber']['unconditional']['trial_mean']
  new_effect_note=f'32个留出来源、188个独立段、5个固定seed的940组配对/1880次生成全部完成，0生成失败；9400数组已独立复核。文本条件平均归一化Huber {c:.6f}，无条件{u:.6f}；文本条件高{(c/u-1)*100:.2f}%，这是重构诊断的负面结果，不能当作TMR或语义结论。'
pp=B/'results/ldm100k_user_handoff_20261028/user_ldm100k_generated_physics_v1/progress.json'
if pp.exists():
 pj=json.loads(pp.read_bytes());pr=pj.get('files',{}).get('receipt',{})
 if pr:
  generated_physics=dict(status=pr['status'],finished=pr.get('finished_trials',0),passed=pr.get('complete_trials',0),failed=pr.get('failed_trials',0),declared=1880,closed_execution_verified=pj.get('closed_execution_verified',False),artifact_audit=pj.get('artifact_audit'))
  new_effect_note+=f" 新生成动作严格回放已结束{generated_physics['finished']}/1880，通过{generated_physics['passed']}，失败{generated_physics['failed']}；闭合核验{generated_physics['closed_execution_verified']}。这与原8278动作筛选分别计数。"
  if not generated_physics['closed_execution_verified']:new_effect_note+=' 进行中按时长排序，当前通过数不能当最终通过率。'
tracking_fidelity=None
tmr_numeric_inputs=None
tmr_evaluator_training=None
ep=B/'results/semantic_evaluator_20261029/tmr69d_evaluator500_train_v1/progress.json'
if ep.exists():
 ej=json.loads(ep.read_bytes());er=ej.get('files',{}).get('receipt',{})
 if er:
  tmr_evaluator_training=dict(status=er['status'],optimizer_updates=er.get('formal_optimizer_updates',0),completed_epochs=er.get('completed_epochs',0),target_epochs=500,expected_optimizer_updates=507500,closed_execution_verified=ej['closed_execution_verified'],generator_teacher_additional_updates=0,semantic_metrics_computed=False)
  new_effect_note+=f" 独立机器人语义评价器当前{tmr_evaluator_training['completed_epochs']}/500轮、{tmr_evaluator_training['optimizer_updates']}次已核验更新；它与已结束的VAE/LDM十万步分别计数，正式语义指标仍待评价器训练和冻结验收。"
np=B/'results/semantic_evaluator_20261029/tmr69d_numeric_data_v1/progress.json'
if np.exists():
 nj=json.loads(np.read_bytes())
 if nj.get('closed_execution_verified'):
  nr=nj['files']['receipt'];tmr_numeric_inputs=dict(receipt_sha256=nj['sha256']['receipt'],source_families=nr['original_train_sources'],source_feature_frames=nr['actual_unique_source_feature_frames'],verified_sample_windows=nr['actual_sample_windows_verified'],source_normalized_RAM_bytes=nr['full_normalized_feature_RAM_bytes'],formal_evaluator_training_updates=0,semantic_metrics_available=False)
  new_effect_note+=' 机器人语义评价器的5361个原TRAIN来源与34068真实文本窗口已数值核验；原VAL未用于梯度，正式评价器尚未训练，文本模型正在单独导入。'
fp=B/'results/ldm100k_user_handoff_20261028/user_ldm100k_tracking_fidelity_v1/progress.json'
if fp.exists():
 fj=json.loads(fp.read_bytes())
 if fj.get('closed_execution_verified'):
  fr=fj['files']['receipt'];tracking_fidelity=dict(aggregate=fr['aggregate'],processed_trials=fr['processed_trials'],receipt_sha256=fj['sha256']['receipt'],artifact_audit=fj['artifact_audit'])
  c=fr['aggregate']['conditional_cfg5']['tracked14'];u=fr['aggregate']['unconditional']['tracked14']
  new_effect_note+=f" 已从保存的真实执行姿态完成1880条跟踪误差核验；完整覆盖且最大逐帧根相对平均误差≤0.3m：条件{c['full_coverage_successes']}/940、无条件{u['full_coverage_successes']}/940；逐trial等权全局MPJPE {c['trial_equal_global_mpjpe_mm']:.2f}/{u['trial_equal_global_mpjpe_mm']:.2f}mm。此补充判据独立于原strict终止规则，无新物理回放或优化更新，正式TMR/在线仍待验收。"
sample=datetime.datetime.fromtimestamp(handoff_observed,datetime.timezone(datetime.timedelta(hours=8))).isoformat()
spec=B/'configs/teacher15000-direct-strict-spec.json';h=hashlib.sha256(spec.read_bytes()).hexdigest()
current=dict(schema='textop.direct_strict_user_progress.v1',run=run,status=r['status'],sample_beijing=sample,
 teacher_stopped=True,frozen_checkpoint_completed=15000,checkpoint_sha256=r['policy_checkpoint_sha256'],
 source_count=8278,finished=done,passed_complete=passed,failed=failed,pending_or_in_progress=pending,
 actual_controls=r.get('actual_vectorized_control_calls',0),actual_physics_steps=r.get('actual_scene_physics_steps',0),
 historical_ema_candidates=0,historical_ema_attempted=8278,historical_ema_above_threshold=8278,historical_ema_unattempted=0,
 direct_replay_bypasses_ema_candidate_gate=True,physical_thresholds_changed=False,optimizer_updates=0,
 completed_execution_verified=audit_verified,raw_light_collector_trace_acceptance=j['completed_execution_verified'],
 strict_physics_accepted=audit_verified,strict_audit_receipt_sha256=audit_sha,handoff_stage=handoff_stage,handoff_stage_label=phase_label,
 actual_unattempted_sources=0 if audit_verified else None,filtered_corpus_accepted=corpus_accepted,generator_started=generator_started,
 generator_primitive_attempts=generator_attempts,vae_target_primitive_attempts=100000 if user_100k_target else 200000,vae_selected_checkpoint_attempts=vae_selected_step,ldm_primitive_attempts=ldm_attempts,ldm_target_primitive_attempts=ldm_target,user_ldm_endpoint_control_status=user_ldm_control_status,user_100k_control_status=user_control_status,handoff_workers=handoff_workers,active_generator_spec_sha256=generator_spec_sha,
 ldm_selected_checkpoint_sha256=ldm_selected_checkpoint_sha256,ldm_successor_run=ldm_successor_run,ldm_successor_dispatch_unconfirmed=ldm_dispatch_unconfirmed,
 ldm_diagnostic_summary=diagnostic_summary,generated_motion_physics=generated_physics,generated_motion_tracking_fidelity=tracking_fidelity,tmr_numeric_inputs=tmr_numeric_inputs,tmr_evaluator_training=tmr_evaluator_training,
 remote_observation_available=not access_unavailable,last_access_attempt_beijing=access_attempt,last_verified_handoff_stage=last_verified_handoff_stage,
 active_spec_sha256=h,worker_pids=[x['pid'] for x in j['workers']],common_lock_free=handoff_lock_free,
 physical_replay_lock_released=j['common_lock_free'],
 runtime_receipt_count_note='replayed_original_sources and unattempted_original_sources describe planned selected/outside scope until execution ends; live pending is8278-finished_trials',
 source_progress_sha256=hashlib.sha256((O/run/'progress.json').read_bytes()).hexdigest())
(O/'current.json').write_text(json.dumps(current,indent=2),encoding='utf-8')
priority=B/'configs/teacher15000-direct-strict-priority-20261027.json';p=json.loads(priority.read_text(encoding='utf-8'))
p.update(status=r['status'],runtime_launch_enabled=not audit_verified,teacher_stopped_verified=True,active_full_run=run,
 active_spec_sha256=h,actual_full_started=r.get('physical_tracking_executed',False),strict_physics_accepted=audit_verified,
 strict_audit_receipt_sha256=audit_sha,handoff_stage=handoff_stage,
 user_asked_to_relax_threshold=True,relaxed_scope='All originals admitted; no numerical physical termination threshold change',
 historical_ema_candidates_are_not_physical_replay_failures=True)
priority.write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
text=f'''# TextOp：停止跟踪器，直接全库物理筛查

采样时间：{sample}。跟踪器已安全停止；固定使用已完整备份的15000轮模型，不再训练跟踪器。

| 正式全库回放 | 数量 |
|---|---:|
| 总范围 | 8278 |
| 已结束 | {done} |
| 完整轨迹通过 | {passed} |
| 触发物理终止而失败 | {failed} |
| 尚未结束（含未测试与执行中） | {pending} |

状态：{r['status']}。{closure_note}
后续阶段：{phase_label}。新生成器此前已正式开训：{generator_started}；VAE目标{100000 if user_100k_target else 200000}步，最后已核验VAE步数{generator_attempts}，已选择检查点{vae_selected_step}，正式LDM步数{ldm_attempts}。
{access_note}
{new_effect_note}
此前两次入口核验测试的是同一组8个短动作，两次都是8/8完整通过；不能当作全库通过率，也不能相加为16个不同动作。

15000轮历史EMA统计另行保留：8278个来源全部有尝试，EMA失败率≤5%候选0个，超过门槛8278个，未尝试0个。这不是标准物理回放的失败数量。
按用户要求，当前放开历史EMA入选限制，全部8278个原始动作均进入固定策略的标准严格完整回放。物理终止阈值没有数值放宽。每个动作一次完整回放，记录通过/失败，不能把单次通过解释成已证明失败概率≤5%。这个直接路线与论文原两阶段候选筛选有差异。

实际回放使用16个环境，GPU1、CPU0–3共4核、Torch显存比例0.65；原SDK辅助GPU0上下文属于历史回放，已随回放作业退出。每次控制对应4次物理步进，保存真实执行姿态与全部失败。当前优化更新为0。
只有新严格动作结果、非空REFERENCE+BABEL语料、源/镜像隔离、TRAIN统计、真实CLIP及加载/梯度/恢复/持久容量验收通过后，才自动启动新筛选语料VAE{100000 if user_100k_target else 200000}步→LDM{ldm_target}步。未筛选8100步VAE继续作为暂停对照保留。

真实15000检查点SHA：`{r['policy_checkpoint_sha256']}`。当前运行：`{run}`；配置SHA：`{h}`。
原生SDK关闭未返回时仍使用已验证的有界退出机制；最终语义、端点质量、可部署WBC、在线质量和视频尚未完成，不能称论文完整效果。
'''
(B/'reports/teacher15000_direct_strict_20261027.md').write_text(text,encoding='utf-8')
dst=D/'TextOp当前物理筛选进度_20261027.md';dst.write_text(text,encoding='utf-8')
body='<!doctype html><meta charset="utf-8"><title>TextOp物理筛选进度</title><style>body{font:17px/1.8 system-ui;max-width:1000px;margin:32px auto;padding:24px;background:#101927;color:#edf2fa}</style>'+''.join('<p>'+html.escape(line)+'</p>' for line in text.splitlines() if line)
(D/'TextOp当前物理筛选进度_20261027.html').write_text(body,encoding='utf-8')
user_goal_note=(f'<p>VAE已完成{generator_attempts:,}步；LDM已正式训练{ldm_attempts:,}/{ldm_target:,}步。当前{html.escape(phase_label)}。</p>' if user_100k_target and ldm_attempts>0 else f'<p>按你的最新要求：VAE目标100,000步，最近已核验{generator_attempts:,}步；保存和验收后自动转入LDM{ldm_target:,}步。当前{html.escape(phase_label)}。</p>' if user_100k_target else '')
if new_effect_note:user_goal_note+='<p>'+html.escape(new_effect_note)+'</p>'
banner=f'<!--direct-strict-current-start--><section style="padding:24px;background:#edf6ee;color:#153a24;font:17px/1.8 system-ui"><h2>最新训练目标与物理筛选结果</h2>{user_goal_note}<p>采样{html.escape(sample)}：完整通过{passed}，失败{failed}，尚未结束{pending}。未测试不算失败。历史EMA候选0与本次物理结果分别记录。</p><a href="TextOp当前物理筛选进度_20261027.html">查看当前完整说明</a></section><!--direct-strict-current-end-->'
for name in ['TextOp下载与训练报告.html','TextOp在线功能验收与完整训练接续_20261026.html']:
 q=D/name
 if q.exists():
  old=q.read_text(encoding='utf-8');old=re.sub(r'<!--direct-strict-current-start-->.*?<!--direct-strict-current-end-->','',old,flags=re.S)
  m=re.search(r'<body[^>]*>',old,re.I);q.write_text(old[:m.end()]+banner+old[m.end():] if m else banner+old,encoding='utf-8')
for q in [O/'current.json',priority,B/'reports/teacher15000_direct_strict_20261027.md']:
 d=D/q.relative_to(B);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,d);assert q.read_bytes()==d.read_bytes()
print(json.dumps(current,ensure_ascii=False))

"""User-corrected filtering-first route takes precedence over historical receipts."""
import html,json,pathlib,re
BASE=pathlib.Path(__file__).resolve().parents[1];D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
p=BASE/'results/generator_pause_20261027/private_backup_verified.json';r=json.loads(p.read_text(encoding='utf-8'))
assert r['status']=='PREFILTER_GENERATOR_PAUSED_CHECKPOINT_PRIVATE_BACKUP_VERIFIED'
loose=BASE/'results/teacher10000_full_filter_20261027/relaxed_scope_v1/summary.json'
l=json.loads(loose.read_text(encoding='utf-8')) if loose.exists() else None
extra='全库筛选尚未完成；下一步读取10k教师的全8278来源宽松阶段统计，再对通过第一阶段的动作做严格完整回放。'
if l:
 c=l['counts'];extra=f"最近已核验的是10000轮检查点的全部{l['source_count']}来源真实宽松训练统计：{c['eligible_for_strict']}条达到严格回放候选门槛，{c['rejected_relaxed_rate']}条宽松历史失败率超过5%，{c['unattempted']}条无有效尝试；这不是15000轮结果，严格回放与最终训练语料验收尚未完成。"
 if c['eligible_for_strict']==0:extra+=' 当前为空候选，不能建立合格语料，也不能据此宣称全部动作不可行。EMA最低18.66%、中位24.65%；原始累计失败率最低13.33%。发布版本统计有初始化与学习历史影响，不据此断言唯一根因。保持5%阈值并改进教师，在新15k检查点复查，不强制等40k才检查。'
recovery=BASE/'results/teacher10000_full_filter_20261027/progress.json'
verified15=False
stop_path=BASE/'results/teacher15000_direct_strict_20261027/control/user_stop_v1_result.json'
stopped=stop_path.exists() and json.loads(stop_path.read_text(encoding='utf-8')).get('common_sdk_lock_released') is True
checkpoint15=BASE/'results/filter_recovery_checkpoints_20261027/round_15000'
if (checkpoint15/'backup.json').exists():
 b15=json.loads((checkpoint15/'backup.json').read_text(encoding='utf-8'))
 if b15.get('status')=='FORMAL_CHECKPOINT_PRIVATE_FULL_AND_GITHUB_INFERENCE_VERIFIED':
  s15=json.loads((checkpoint15/'relaxed_summary.json').read_text(encoding='utf-8'))
  assert b15['private_full_two_location_sha_verified'] and b15['private_relaxed_statistics_two_location_sha_verified']
  assert s15['completed_learning_iterations']==15000 and s15['source_count']==8278
  assert s15['checkpoint_sha256']==b15['audit']['full_checkpoint']['sha256']
  c15=s15['counts']; dist=s15['ema_failure_distribution'];verified15=True
  extra=f"15000轮真实检查点已完成CPU全状态核验及服务器、D盘私有SHA备份，累计300000次优化更新。全部8278条来源均有真实尝试：{8278-c15['unattempted']}条已尝试，{c15['eligible_for_strict']}条达到EMA失败率≤5%的宽松候选门槛，{c15['above_relaxed_threshold']}条超过门槛，{c15['unattempted']}条未尝试，{c15['invalid_denominator']}条统计分母无效。累计{ s15['raw_attempts_total']:,}次尝试；EMA失败率最低{dist['minimum']:.2%}、中位{dist['median']:.2%}。这只是保存的宽松训练历史统计，不是新的严格物理回放。"
  if c15['eligible_for_strict']==0:
   extra+=' 与10k的最低18.66%、中位24.65%相比有所下降。用户随后明确要求停止训练跟踪器、直接物理筛选：用冻结15000轮模型回放全部8278个原始动作，保留原EMA门槛结果，不把它改写成通过。这是用户指定的直接严格回放路线，与论文先按EMA选候选的步骤有区别。'
if stopped:
 extra+=' 跟踪器已实际受控退出，原进程已回收，共同GPU/SDK锁已释放；停止前日志收据15300轮，安全可恢复检查点为15000轮。不会再训练跟踪器。'
 full_progress=BASE/'results/teacher15000_direct_strict_20261027/teacher15000_direct_full_v1/progress.json'
 if full_progress.exists():
  fp=json.loads(full_progress.read_text(encoding='utf-8'));fr=fp.get('files',{}).get('receipt',{})
  extra+=f" 新全库严格回放状态：{fr.get('status')}；已结束{fr.get('finished_trials',0)}/8278，完整通过{fr.get('complete_trials',0)}，物理终止失败{fr.get('failed_trials',0)}。采样时间Unix {fp.get('observed_unix')}；进行中数量不是最终筛选结果。"
 else:extra+=' 新完整回放入口正在接受实际运行核验，尚不能报告全库通过数量。'
elif recovery.exists():
 j=json.loads(recovery.read_text(encoding='utf-8'));tr=j.get('receipt') or {};extra+=f" 当前教师接续状态{tr.get('status')}，精确状态恢复核验{tr.get('resume_exact_state_verified')}，收据轮数{tr.get('completed_learning_iterations')}，原生日志最新完成轮数{j.get('observed_completed_from_native_log')}；每100轮刷新收据，初始0不代表从0重训。"
checkpoint_tools=BASE/'results/filter_recovery_checkpoint_tools_20261027/activation.json'
if checkpoint_tools.exists() and not verified15:
 t=json.loads(checkpoint_tools.read_text(encoding='utf-8'))
 extra+=' 新运行的15k及后续检查点已准备独立备份和全源宽松统计入口，元数据/计数检查及独立文档核查已完成；实际新检查点加载、备份和统计仍等待真实保存，尚未进行严格筛选。工具不停止训练，不复用旧5k/10k来源，不公开完整优化器或逐动作记录。'
handoff=BASE/'configs/teacher15000-user-handoff-plan-20261027.json'
if handoff.exists():
 h=json.loads(handoff.read_text(encoding='utf-8'))
 assert h['target_checkpoint_completed_iterations']==15000 and h['relaxed_failure_rate_maximum']==.05
 extra+=' 用户明确要求：报告实际筛选数量，直接全库严格回放及非空运动—文本语料、CLIP、恢复与容量验收通过后自动启动新的筛选语料VAE200k→LDM300k，无需再次确认。'
physical_complete=False
current_path=BASE/'results/teacher15000_direct_strict_20261027/current.json'
if current_path.exists():
 actual=json.loads(current_path.read_text(encoding='utf-8'))
 physical_complete=actual.get('strict_physics_accepted') is True
 if physical_complete:
  assert actual['finished']==actual['source_count']==8278 and actual['actual_unattempted_sources']==0
  extra+=f" 最新全库最终物理结果已由独立CPU完整性核验接受：8278个全部实测，{actual['passed_complete']}个完整通过，{actual['failed']}个失败，0个未尝试。原SDK及CPU核验作业均退出。后续阶段{actual.get('handoff_stage_label',actual['handoff_stage'])}，新生成器正式开训标志{actual['generator_started']}。固定15k、每源一次、跳过历史EMA候选的限制保留；单次通过不等于失败概率≤5%。"
if physical_complete and actual.get('vae_target_primitive_attempts')==100000:
 extra+=' 用户最新明确将VAE目标缩短为100000步，随后自动训练LDM300000步；旧200k描述仅为历史计划。新控制/质量/导出/LDM阶段以最新真实收据为准，保留原退出事实和100k完整状态。'
text=f'''# TextOp：恢复先筛选、后训练的论文顺序

用户2026-10-27明确调整：安全保存并暂停未筛选数据生成器，先完成全库物理筛选。先前推荐的“未筛选全源先训练”属于额外对照，并非论文主线；现在已取消其自动接续，不能把这条路线说成论文训练数据已合格。

此前未筛选对照生成器已在完整批次边界保存并退出：{r['completed_attempts']}次primitive尝试、{r['optimizer_updates']}次实际Adam更新、{r['skips']}次跳过。完整模型、优化器、训练时钟和随机状态保留，CPU模型/Adam重载一致，CPU随机状态恢复通过；CUDA随机状态完整保存并检查结构，本轮未执行GPU恢复训练。检查点{r['bytes']}字节，SHA `{r['sha256']}`，服务器与D盘私有副本已核验。原始训练输出的非零退出和旧RUNNING收据均保留，独立暂停收据解释为用户授权的完整批次保存退出，不是200k训练完成。

暂停没有向真实训练进程发信号，也没有修改训练源码/参数。独立验证后的临时记录路径标记触发原程序已有紧急保存逻辑；真实检查点落盘、原进程退出、记录解压完整和共同GPU/SDK锁释放均有证据。先前信号方案的独立审查阻断与全部失败记录保留，未用于真实训练。

{extra}

最新接续顺序：停止跟踪器 → 冻结15000轮模型直接对全8278来源严格完整回放 → 保留可跟踪的原始参考动作并配对BABEL文本 → 源隔离、TRAIN-only统计、真实CLIP和加载/恢复/容量验收 → 新筛选语料上的VAE{actual.get("vae_target_primitive_attempts",200000) if physical_complete else 200000}步与LDM300000步。直接回放不宣称历史EMA门槛已通过，也不自动恢复跟踪器训练。未尝试/失败原资产不删除，早期教师失败不证明动作绝对不可行。

原8278来源预处理及私有备份已经完成。当前未筛选VAE只保留为独立对照，不自动恢复，也不自动启动其LDM。完整语义质量、最终WBC/生成增强、在线执行与论文完整效果仍未通过。

论文依据：[TextOp §III-B及附录A-A](https://arxiv.org/html/2602.07439v1)：先用特权跟踪器排除难以可靠跟踪的运动，再将保留下来的AMASS运动与BABEL文本组合训练生成器；宽松阶段失败率超过5%的运动剔除，再按标准终止条件检查其余运动。
'''
(BASE/'reports/filter_first_restored_20261027.md').write_text(text,encoding='utf-8')
body='<!doctype html><meta charset="utf-8"><title>TextOp恢复论文筛选顺序</title><style>body{font:17px/1.8 system-ui;max-width:1020px;margin:40px auto;padding:24px;background:#101927;color:#edf2fa}a{color:#7fc4ff}</style>'+''.join('<h1>'+html.escape(x[2:])+'</h1>' if x.startswith('# ') else '<p>'+html.escape(x)+'</p>' for x in text.splitlines() if x)
name='TextOp恢复论文筛选顺序_20261027.html';(D/name).write_text(body,encoding='utf-8');(D/'TextOp恢复论文筛选顺序_20261027.md').write_text(text,encoding='utf-8')
main=D/'TextOp下载与训练报告.html'
if main.exists():
 old=main.read_text(encoding='utf-8');old=re.sub(r'<!--generator8278-current-start-->.*?<!--generator8278-current-end-->','',old,flags=re.S);old=re.sub(r'<!--filter-first-current-start-->.*?<!--filter-first-current-end-->','',old,flags=re.S)
 headline='15000轮筛选统计已核验，生成器等待合格语料' if verified15 else '生成器已保存暂停，恢复先筛选后训练'
 current=f"15000轮：{c15['eligible_for_strict']}/8278条达到EMA失败率≤5%候选门槛，严格筛选和合格语料仍未完成。" if verified15 else str(r['completed_attempts'])+'步完整状态已在服务器与D盘核验。全库物理筛选尚未完成；未筛选实验不自动接续。'
 if physical_complete:
  headline='全库物理筛查已核验，自动接续筛后生成器'
  current=f"8278个全部实测：完整通过{actual['passed_complete']}，失败{actual['failed']}，未尝试0。当前{actual.get('handoff_stage_label',actual['handoff_stage'])}；生成器正式开训标志{actual['generator_started']}。"
 section='<!--filter-first-current-start--><section style="padding:24px;background:#edf6ee;color:#153a24;font:17px/1.8 system-ui"><h2>最新：'+headline+'</h2><p>'+current+'</p><a href="'+name+'">查看检查点、筛选统计及接续说明</a></section><!--filter-first-current-end-->'
 match=re.search(r'<body[^>]*>',old,re.I);old=old[:match.end()]+section+old[match.end():] if match else section+old;main.write_text(old,encoding='utf-8')
print(json.dumps({'status':r['status'],'steps':r['completed_attempts'],'report':str(D/name)},ensure_ascii=False))

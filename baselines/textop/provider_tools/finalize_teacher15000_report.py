"""Publish only aggregate evidence after the real 15k backup has completed."""
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

B = pathlib.Path(__file__).resolve().parents[1]
D = pathlib.Path('D:/具身智能/TextOp复现_20261024')
F = B / 'results/filter_recovery_checkpoints_20261027/round_15000'
def read(name):
    return json.loads((F/name).read_text(encoding='utf-8'))
backup, stats, comparison = read('backup.json'), read('relaxed_summary.json'), read('saved_counter_comparison.json')
assert backup['status'] == 'FORMAL_CHECKPOINT_PRIVATE_FULL_AND_GITHUB_INFERENCE_VERIFIED'
assert all(backup[k] for k in ['private_full_two_location_sha_verified', 'private_relaxed_statistics_two_location_sha_verified', 'inference_three_location_sha_verified'])
assert stats['completed_learning_iterations'] == 15000 and stats['actual_optimizer_updates'] == 300000
assert stats['source_count'] == 8278 and stats['counts'] == {'eligible_for_strict': 0, 'above_relaxed_threshold': 8278, 'unattempted': 0, 'invalid_denominator': 0}
assert comparison['source_identity_pairs_verified'] == 8278 and comparison['actual_ema_gate_candidates_at_15000'] == 0
audit = read('cpu_audit.json')
assert audit['execution']['child_exit_code'] == 0 and audit['execution']['child_reaped'] and audit['execution']['supervision_passed']
assert audit['audit']['optimizer_step_min'] == audit['audit']['optimizer_step_max'] == 300000
checkpoint = backup['audit']['full_checkpoint']
assert checkpoint['sha256'] == stats['checkpoint_sha256']
stop=json.loads((B/'results/teacher15000_direct_strict_20261027/control/user_stop_v1_result.json').read_text(encoding='utf-8'))
assert stop['common_sdk_lock_released'] and stop['recoverable_completed']==15000
result = dict(schema='textop.teacher15000_result.v1', status='REAL_15000_BACKUP_VERIFIED_ZERO_RELAXED_CANDIDATES',
    completed_iterations=15000, optimizer_updates=300000, checkpoint=checkpoint, source_count=8278,
    attempted_sources=8278, counts=stats['counts'], relaxed_failure_rate_maximum=.05,
    private_backup_verified=True, inference_github_verified=True, cpu_audit_actual_exit_code=0,
    actual_strict_replay_performed=False, filtered_corpus_accepted=False, generator_started=False,
    prefilter_generator_remains_paused=True, threshold_changed=False, teacher_stopped_verified=True,
    next_checkpoint_review=None, next_action='Teacher stopped by explicit user request. Direct strict complete replay of all8278 originals using frozen15000, then accepted corpus and new generator training.',
    notification_prepared=True, notification_delivery_recorded=False,
    sample_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
out=F/'user_result.json'
assert not out.exists()
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
report=f'''# TextOp：15000轮检查点与筛选结果（2026-10-27）

真实15000轮、300000次优化更新已经完成，检查点、CPU全状态审计、服务器与D盘私有备份及公开推理导出摘要均已核验。按用户最新要求，跟踪器已经安全停止，共同GPU/SDK锁已释放；未筛选VAE仍暂停在8100步。

| 项目 | 结果 |
|---|---:|
| 原始动作来源总数 | 8278 |
| 有真实训练尝试的来源 | 8278 |
| EMA失败率≤5%的宽松候选 | 0 |
| 超过宽松门槛 | 8278 |
| 未尝试 / 无效统计 | 0 / 0 |
| 累计尝试 | 21,785,568 |

这里的0是宽松历史EMA门槛结果，不能解释成全部动作绝对不可行。用户随后明确要求直接物理筛选，因此将用固定15000轮模型对全部8278个原始动作进行严格完整回放，以实际通过动作建立生成器语料。这是用户指定的直接回放路线，与论文先按历史EMA筛选候选的步骤有区别。完整物理结果和语料验收尚未完成，生成器尚未启动。原始动作、全部失败记录和检查点均保留。

## 与10000轮对比

| 统计 | 10000轮 | 15000轮 |
|---|---:|---:|
| EMA失败率最低值 | 18.6609% | 11.7442% |
| EMA失败率中位数 | 24.6525% | 16.0868% |
| EMA失败率95分位 | 52.6600% | 49.3946% |
| ≤5%宽松候选 | 0 | 0 |

对两次既有逐动作统计文件按来源、顺序、参考姿态SHA、帧数和split逐项配对，8278条全部匹配。8083条EMA下降，195条上升。10k至15k新增5,898,261次尝试，其中275,700次失败，合计失败率4.6743%；7645条来源在这一新增区间的原始失败率≤5%。**这些只是训练期间的区间诊断，不能替代规定的逐动作历史EMA筛选，不能把7645写成已通过筛选。** 区间中策略持续变化，没有进行固定策略重评估，也没有证明初始化或学习历史是唯一原因。

## 核验与保存

- 完整检查点：{checkpoint['bytes']}字节，SHA `{checkpoint['sha256']}`。
- 可信CPU加载审计：真实退出0，80个有限张量，所有Adam步数300000；公开网络与归一化导出25个张量精确重载一致。没有新增GPU恢复训练或物理试验。
- 完整状态及逐动作决定保持服务器、D盘私有；[公开推理权重]({backup['github_asset']})仅含网络、归一化和聚合元数据，GitHub资产摘要已核验。
- 完整备份和汇总证据：`results/filter_recovery_checkpoints_20261027/round_15000/backup.json`；统计：`relaxed_summary.json`；区间比较：`saved_counter_comparison.json`。

## 接续安排

跟踪器已经实际退出：原SDK进程通过绑定原身份的一次SIGINT停止，子进程退出0且已回收，外层退出1如实保留，因为没有完成旧40k目标。最后训练收据15300轮，安全可恢复检查点为15000轮；不宣称15300轮内存状态已保存。不会再训练跟踪器。新的独立入口绑定当前15k和停机收据，先验证实际回放，再启动全8278来源的标准严格完整回放。保留5%统计定义和原严格终止阈值；完成实际回放及非空运动—文本语料、源隔离、TRAIN统计、CLIP、加载/梯度/恢复和容量验收后，自动启动新筛选语料VAE200k→LDM300k。不会恢复未筛选8100步模型或训练空集。

最终语义质量、可部署WBC、生成增强、实际在线执行与视频仍未完成；本结果不是论文完整复现验收。
'''
(B/'reports/teacher15000_filter_result_20261027.md').write_text(report,encoding='utf-8')
current='''## CURRENT — real15000 verified: ZERO relaxed candidates (2026-10-27)

Read reports/teacher15000_filter_result_20261027.md and
results/filter_recovery_checkpoints_20261027/round_15000/user_result.json.
Actual15000/300000 checkpoint90799682B SHA66e87c9471d2860f1b43a18a41e938c12647c93212f1fe44c8be1f1b27febc88:
CPUaudit actualexit0/80finite/Adam300000/25inference tensors exact;
server+D privatefullstate/decisions and GitHub inference digest verified.
8278total/8278attempted/0EMA<=5%candidates/8278above/0unattempted/0invalid.
EMAmin11.7442%,median16.0868%; sourcepaired8083lower/195higher versus10k.
New10k-to15k raw interval7645<=5% is DIAGNOSTIC ONLY, not candidates or strict
acceptance; current policy was learning, not a fixed-policy reassessment.
The user explicitly ordered STOP teacher and direct physical filtering.
Teacher now actually stopped, original child reaped, common SDK lock released.
Last receipt15300 is not a saved checkpoint; freeze actual15000 for replay.
Direct strict all8278 originals bypasses historical EMA selection by explicit
user choice; preserve EMA0 and disclose this departure from paper selection.
Read TEACHER15000_DIRECT_STRICT_ENVIRONMENT.md and new direct-priority spec.
Do not repeat15k audit/export/backups/comparison/release or old diagnostics.
New strict runtime acceptance precedes full8278 launch and corpus acceptance;
prefilter8100 remains paused, generator not yet started.
Never train empty or unfiltered data or silently change5%/strict thresholds.
One identity-bound SIGINT stopped the teacher after backup. Do not repeat
stop/backup or resume teacher. Fullpaper still pending.
This CURRENT overrides all historical pending15k/15minute status below.

'''
for name in ['CONTINUATION.md','README.md']:
    p=B/name
    old=p.read_text(encoding='utf-8')
    assert not old.startswith('## CURRENT — real15000 verified:')
    p.write_text(current+old,encoding='utf-8')
for src in [out,B/'reports/teacher15000_filter_result_20261027.md']:
    dest=D/src.relative_to(B);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(src.read_bytes())
    assert hashlib.sha256(dest.read_bytes()).digest()==hashlib.sha256(src.read_bytes()).digest()
subprocess.run([sys.executable,str(B/'provider_tools/collect_filter_first_report.py')],check=True)
print(json.dumps({'status':result['status'],'report':'reports/teacher15000_filter_result_20261027.md','generator_started':False}))

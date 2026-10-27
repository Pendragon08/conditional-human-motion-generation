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
sample=datetime.datetime.fromtimestamp(j['observed_unix'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
spec=B/'configs/teacher15000-direct-strict-spec.json';h=hashlib.sha256(spec.read_bytes()).hexdigest()
current=dict(schema='textop.direct_strict_user_progress.v1',run=run,status=r['status'],sample_beijing=sample,
 teacher_stopped=True,frozen_checkpoint_completed=15000,checkpoint_sha256=r['policy_checkpoint_sha256'],
 source_count=8278,finished=done,passed_complete=passed,failed=failed,pending_or_in_progress=pending,
 actual_controls=r.get('actual_vectorized_control_calls',0),actual_physics_steps=r.get('actual_scene_physics_steps',0),
 historical_ema_candidates=0,historical_ema_attempted=8278,historical_ema_above_threshold=8278,historical_ema_unattempted=0,
 direct_replay_bypasses_ema_candidate_gate=True,physical_thresholds_changed=False,optimizer_updates=0,
 completed_execution_verified=j['completed_execution_verified'],filtered_corpus_accepted=False,generator_started=False,
 active_spec_sha256=h,worker_pids=[x['pid'] for x in j['workers']],common_lock_free=j['common_lock_free'],
 runtime_receipt_count_note='replayed_original_sources and unattempted_original_sources describe planned selected/outside scope until execution ends; live pending is8278-finished_trials',
 source_progress_sha256=hashlib.sha256((O/run/'progress.json').read_bytes()).hexdigest())
(O/'current.json').write_text(json.dumps(current,indent=2),encoding='utf-8')
priority=B/'configs/teacher15000-direct-strict-priority-20261027.json';p=json.loads(priority.read_text(encoding='utf-8'))
p.update(status=r['status'],runtime_launch_enabled=True,teacher_stopped_verified=True,active_full_run=run,
 active_spec_sha256=h,actual_full_started=r.get('physical_tracking_executed',False),strict_physics_accepted=False,
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

状态：{r['status']}。进行中统计不是最终筛选结论。未测试动作没有被过滤。
此前两次入口核验测试的是同一组8个短动作，两次都是8/8完整通过；不能当作全库通过率，也不能相加为16个不同动作。

15000轮历史EMA统计另行保留：8278个来源全部有尝试，EMA失败率≤5%候选0个，超过门槛8278个，未尝试0个。这不是标准物理回放的失败数量。
按用户要求，当前放开历史EMA入选限制，全部8278个原始动作均进入固定策略的标准严格完整回放。物理终止阈值没有数值放宽。每个动作一次完整回放，记录通过/失败，不能把单次通过解释成已证明失败概率≤5%。这个直接路线与论文原两阶段候选筛选有差异。

实际回放使用16个环境，GPU1、CPU0–3共4核、Torch显存比例0.65；SDK辅助GPU0上下文仍存在。每次控制对应4次物理步进，保存真实执行姿态与全部失败。当前优化更新为0。
只有新严格动作结果、非空REFERENCE+BABEL语料、源/镜像隔离、TRAIN统计、真实CLIP及加载/梯度/恢复/持久容量验收通过后，才自动启动新筛选语料VAE200k→LDM300k。未筛选8100步VAE继续作为暂停对照保留。

真实15000检查点SHA：`{r['policy_checkpoint_sha256']}`。当前运行：`{run}`；配置SHA：`{h}`。
原生SDK关闭未返回时仍使用已验证的有界退出机制；最终语义、端点质量、可部署WBC、在线质量和视频尚未完成，不能称论文完整效果。
'''
(B/'reports/teacher15000_direct_strict_20261027.md').write_text(text,encoding='utf-8')
dst=D/'TextOp当前物理筛选进度_20261027.md';dst.write_text(text,encoding='utf-8')
body='<!doctype html><meta charset="utf-8"><title>TextOp物理筛选进度</title><style>body{font:17px/1.8 system-ui;max-width:1000px;margin:32px auto;padding:24px;background:#101927;color:#edf2fa}</style>'+''.join('<p>'+html.escape(line)+'</p>' for line in text.splitlines() if line)
(D/'TextOp当前物理筛选进度_20261027.html').write_text(body,encoding='utf-8')
banner=f'<!--direct-strict-current-start--><section style="padding:24px;background:#edf6ee;color:#153a24;font:17px/1.8 system-ui"><h2>最新：跟踪器已停止，全库8278个动作物理筛查</h2><p>采样{html.escape(sample)}：完整通过{passed}，失败{failed}，尚未结束{pending}。未测试不算失败。历史EMA候选0与本次物理结果分别记录。</p><a href="TextOp当前物理筛选进度_20261027.html">查看当前完整说明</a></section><!--direct-strict-current-end-->'
for name in ['TextOp下载与训练报告.html','TextOp在线功能验收与完整训练接续_20261026.html']:
 q=D/name
 if q.exists():
  old=q.read_text(encoding='utf-8');old=re.sub(r'<!--direct-strict-current-start-->.*?<!--direct-strict-current-end-->','',old,flags=re.S)
  m=re.search(r'<body[^>]*>',old,re.I);q.write_text(old[:m.end()]+banner+old[m.end():] if m else banner+old,encoding='utf-8')
for q in [O/'current.json',priority,B/'reports/teacher15000_direct_strict_20261027.md']:
 d=D/q.relative_to(B);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,d);assert q.read_bytes()==d.read_bytes()
print(json.dumps(current,ensure_ascii=False))

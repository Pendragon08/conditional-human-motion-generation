"""Archive an observed stopped run; no SSH dispatch, tuning, or physics replay."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / 'results/world_residual_ppo_20261007'
FEEDBACK = ROOT / 'results/feedback_paused_isaac_20261007'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def latest(directory, pattern):
    return max(directory.glob(pattern), key=lambda path: path.stat().st_mtime)


def main():
    status_path = latest(WORLD / 'raw/monitoring', 'progress_*.json')
    feedback_path = latest(FEEDBACK, 'monitoring_*.json')
    evidence_path = latest(FEEDBACK, 'failure_evidence_*.json')
    status, feedback, evidence = map(read, [status_path, feedback_path, evidence_path])
    world_state = json.loads(status['world']['text'])
    feedback_state = json.loads(status['feedback']['text'])
    assert world_state['status'] == feedback_state['status'] == 'FAILED'
    assert evidence['world_progress']['completed_update'] == 41
    assert not feedback['active_processes']
    assert all(all(values.values()) for key, values in evidence.items() if key.startswith('configs/'))
    observed = feedback['observed_unix']
    date = datetime.fromtimestamp(observed, ZoneInfo('Asia/Shanghai')).isoformat()

    partial = read(WORLD / 'partial_control_results.json')
    assert partial['completed_train_runs'] == partial['completed_eval_runs'] == 13
    receipt_path = ROOT / partial['raw_receipt']
    receipt = read(receipt_path)
    compact = []
    for record in receipt['completed_records']:
        summary = dict(record['summary'])
        summary.pop('audit', None)
        item = {key: value for key, value in record.items() if key not in ['summary', 'update_rows']}
        item['summary'] = summary
        if 'update_rows' in record:
            item['update_rows_count'] = len(record['update_rows'])
            item['last_update'] = record['update_rows'][-1]
        compact.append(item)
    compact_path = WORLD / 'raw/monitoring/completed_compact_20261008.json'
    write(compact_path, {
        'observed_beijing': date, 'queue': world_state, 'completed_records': compact,
        'full_local_receipt': receipt_path.relative_to(ROOT).as_posix(),
        'full_local_receipt_sha256': sha(receipt_path),
        'repeated_per_update_rows_omitted_from_github_copy': True,
        'full_trace_independent_audit': False,
    })
    partial.update({
        'status': 'PARTIAL_QUEUE_FAILED_NO_FINAL_GATE_DECISION',
        'queue_status': 'FAILED', 'currently_training': False,
        'interrupted_arm': 'W-correct', 'interrupted_seed': 44,
        'last_written_completed_update': 41, 'last_written_actual_transitions': 8060928,
        'failed_stage_child_exit': 0, 'failed_stage_wrapper_exit': 1,
        'completion_sentinel_missing': True, 'final_checkpoint_missing': True,
        'specific_early_exit_root_cause_established': False,
        'shuffled_seed44_started': False,
        'github_compact_receipt': compact_path.relative_to(ROOT).as_posix(),
        'github_compact_receipt_sha256': sha(compact_path),
    })
    write(WORLD / 'partial_control_results.json', partial)
    progress_path = WORLD / 'control_progress.json'
    progress = read(progress_path)
    progress.update({
        'formal_training_status': 'FAILED · 已完成13/15组训练与评估；W-correct seed44停在41/500',
        'completed_train_runs': 13, 'completed_eval_runs': 13,
        'currently_training': False, 'innovation_validated': False,
    })
    write(progress_path, progress)

    summaries, rows, receipts = [], [], []
    for seed in [42, 43]:
        entry = feedback[f'feedback_seed{seed}']['summary.json']
        s = entry['data']
        assert s['status'] == 'COMPLETE' and s['steps'] == 500 and s['num_envs'] == 132
        assert s['PPO_updates'] == 0 and s['no_extra_IK'] and s['same_episode_branch_without_state_reset']
        assert evidence[f'feedback_physics_{seed}']['sentinels'] == 1
        assert len(s['rows']) == 132
        assert all(math.isfinite(x['post_branch_original_joint_rmse_auc_rad_s']) for x in s['rows'])
        summaries.append(s)
        rows.extend({**row, 'seed': seed} for row in s['rows'])
        receipts.append({
            'seed': seed, 'summary_sha256': entry['sha256'],
            'rollout_sha256': s['rollout_sha256'], 'physics_log_sha256': evidence[f'feedback_physics_{seed}']['sha256'],
            'completion_sentinel_count': 1, 'paused_wall_s': s['paused_wall_s'],
            'candidate_audits': s['candidate_audits'],
            'exit_evidence': 'coordinator accepted returncode==0 before persisting seeds_completed; no separate child exit receipt',
        })

    estimates = {}
    for arm in ['fixed', 'repeat', 'planned', 'alignment', 'actual', 'shuffled']:
        selected = [row for row in rows if row['arm'] == arm]
        push = [row for row in selected if row['domain'] == 'push']
        clean = [row for row in selected if row['domain'] == 'clean']
        estimates[arm] = {
            'attempted': len(selected),
            'goal_hits': sum(row['virtual_goal_success'] for row in selected),
            'virtual_goal_success_rate': statistics.mean(row['virtual_goal_success'] for row in selected),
            'missing_retargets': sum(row['retarget_missing'] for row in selected),
            'first_falls': sum(row['first_fall_step'] is not None for row in selected),
            'failures_including_missing': sum(row['failure'] for row in selected),
            'push_original_tracking_auc_rad_s': statistics.mean(row['post_branch_original_joint_rmse_auc_rad_s'] for row in push),
            'clean_original_joint_rmse_rad': statistics.mean(row['whole_episode_original_joint_rmse_rad'] for row in clean),
            'push_AUC_by_seed': {
                str(seed): statistics.mean(row['post_branch_original_joint_rmse_auc_rad_s'] for row in push if row['seed'] == seed)
                for seed in [42, 43]
            },
        }
        assert len(selected) == 44
    result = {
        'status': 'PARTIAL_QUEUE_FAILED_NO_FINAL_GATE_DECISION', 'observed_beijing': date,
        'queue_status': 'FAILED', 'completed_physical_seeds': [42, 43],
        'planned_physical_seeds': [42, 43, 44], 'completed_attempted_episodes': 264,
        'planned_attempted_episodes': 396, 'generated_human_samples': 198,
        'all_nine_generation_stages_complete': True, 'no_formal_PPO_in_this_test': True,
        'failed_seed': 44,
        'primary_native_failure': 'assert finite and float(excess.max())<=1e-4 at candidate loading, before continuation',
        'candidate_nonfinite_vs_limit_excess_separately_identified': False,
        'coordinator_secondary_failure': feedback_state['reason'],
        'failed_seed_rollout_and_summary_missing': True,
        'failed_seed_not_imputed_as_completed_zero_score': True,
        'estimates_completed_seeds_only': estimates, 'attempted_denominator_retained_in_completed_seeds': True,
        'completed_seed_receipts': receipts,
        'generation_wall_s_by_seed_arm': {
            str(seed): {arm: feedback[f'feedback_seed{seed}'][f'generation_{arm}_timing.json']['wall_s']
                        for arm in ['planned', 'actual', 'shuffled']}
            for seed in [42, 43, 44]
        },
        'partial_descriptive_estimates_only': True, 'full_trace_independent_audit': False,
        'three_seed_gate_evaluated': False, 'innovation_validated': False,
        'real_time_control': False, 'physical_entity_contact_evaluated': False,
        'randomized_block_not_exact_PhysX_clone': True,
        'same_current_Isaac_episode_context': True, 'extra_IK': False,
        'protected_source_checks': {key: value for key, value in evidence.items() if key.startswith('configs/')},
        'rows': rows,
    }
    write(FEEDBACK / 'partial_results_20261008.json', result)
    failure = {
        'observed_beijing': date, 'world_queue': world_state, 'feedback_queue': feedback_state,
        'failed_world_stage_state': feedback['train_W-correct_44.state'],
        'failed_world_log_sha256': feedback['train_W-correct_44.log']['sha256'],
        'world_last_written_update': evidence['world_progress'],
        'world_final_checkpoint_exists': False, 'world_completion_sentinel_exists': False,
        'world_four_minute_trace_is_periodic_diagnostic_not_wrapper_timeout': True,
        'world_stop_cause_not_proven': True,
        'feedback_seed44_physics_log': evidence['feedback_physics_44'],
        'no_active_world_or_feedback_processes': True,
        'gpu_snapshot': status['gpu'], 'disk_snapshot': status['disk'],
        'source_hashes_still_match': result['protected_source_checks'],
        'telemetry_last_training_unix': json.loads(status['viewer_meta']['text'])['telemetry_wall_time'],
        'renderer_capture_timestamp_is_not_current_training_timestamp': True,
        'queue_restart_or_source_changes_this_turn': False,
    }
    write(FEEDBACK / 'failure_status_20261008.json', failure)

    section = [f'\n## {date}：队列停止与已完成结果\n',
        '当前完成13/15组训练及13/15组固定最终评估。seed42和43五臂齐全，seed44三个PPO对照完成；W-correct seed44最后记录41/500更新、8,060,928转移后提前结束，未产生最终检查点、summary或完成哨兵。stage记录child_exit=0，但wrapper_exit=1，必须记FAILED，不能以退出0冒充完成。W-shuffled-original seed44未启动。日志中的4分钟Timeout文字是预设重复栈诊断；栈后仍记录update41，不是已证实的训练超时或故障根因。未发现可据以断言OOM的日志，具体提前退出原因尚待诊断。原输出全部保留，未重启或变更冻结协议。\n',
        '|种子|臂|首回合成功|首回合失效|受扰RMSE积分 rad·s↓|无扰RMSE rad↓|\n|---|---|---:|---:|---:|---:|']
    for row in partial['rows']:
        section.append(f"|{row['seed']}|{row['arm']}|{row['first_episode_success']:.1%}|{row['first_episode_failure']:.2%}|{row['recovery_RMSE_AUC_rad_s']:.6f}|{row['clean_recovery_RMSE_rad']:.6f}|")
    section.extend(['\nW-correct相对P0：seed42恢复积分变差11.34%，seed43改善15.55%；相对有序历史对照分别变差24.70%、改善12.98%；相对错配监督分别变差34.03%、改善11.45%。种子间方向不一致，未据局部正结果宣布有效。13组本协议的首回合与任意reset成功均0，必须先审计参考、初始状态、奖励与成功计数的接口一致性，不能仅凭误差下降证明任务完成，也不与旧85.3%比较。\n',
        '已逐组检查500条连续更新、正常完成哨兵、最终模型SHA与评估源SHA、11参考宏统计；本次没有独立重建全部正式轨迹几何。正式三种子控制门未齐、未判定，未导出核验评估视频。结构化结果为partial_control_results.json；GitHub保留去除重复逐轮记录的completed_compact_20261008.json，完整20MB读取回执仍在本地，哈希与路径均记录。\n',
        '服务器检查时无此队列计算进程；GPU0约46.7GiB、GPU1约38.8GiB显存空余。6013渲染器持续截图并不代表训练继续，实际telemetry停留在提前退出前的最后姿态。下一步先定位退出与0成功接口问题，再决定如何保留原失败并补齐缺失种子；此轮未自动重复训练。\n'])
    with (ROOT/'reports/27_World_model_PPO_runtime_control_ablation_20261007.md').open('a',encoding='utf-8') as stream:
        stream.write('\n'.join(section))
    section = [f'\n## {date}：执行反馈完成两种子，第三种子安全检查中止\n',
        'seed42/43各132环境×500步，264个完整尝试回合；三个种子的9批扩散生成共198条人体动作已完成，每批22条耗时193.39—195.21秒。seed44在载入候选参考时触发有限性/关节硬限位联合断言，尚未单独确认是非有限值还是限位超差。未写rollout/summary，不能记完成或人为补0分。协调器随后读取缺失summary而报FileNotFoundError，是次生错误，原生AssertionError已保全。\n',
        '|臂|完成回合|虚拟目标命中|缺失重定向|首次跌倒|受扰原参考RMSE积分 rad·s↓|干净原参考RMSE rad↓|\n|---|---:|---:|---:|---:|---:|---:|']
    for arm, value in estimates.items():
        section.append(f"|{arm}|{value['attempted']}|{value['goal_hits']}/44 ({value['virtual_goal_success_rate']:.1%})|{value['missing_retargets']}|{value['first_falls']}|{value['push_original_tracking_auc_rad_s']:.6f}|{value['clean_original_joint_rmse_rad']:.6f}|")
    section.extend(['\n表格仅汇总已完成的seed42/43，生成被拒样本保留分母，跟踪按冻结规则使用2rad惩罚；首次跌倒与缺失重定向分开记录，不能把拒绝等同物理跌倒。真实反馈actual只有9/44虚拟目标命中，冻结原参考fixed为32/44，计划重生成planned为8/44；actual仅32/44参考通过重定向。局部结果未显示可用于论文创新主张的收益，尤其不能绕过简单对齐/原参考对照。完整三种子门未评，未重抽样、改阈值或重跑。\n',
        '同次Isaac真实执行前缀与暂停后续接已产生成绩，但这仍是离线随机区组试验、零PPO更新，不能冒充实时CLoSD式闭环。两完整种子的生成/重定向暂停共21.54与21.22分钟；稀疏人机桥、非精确跨臂PhysX克隆、虚拟脚踝目标及单一Kick语义限制继续保留。独立完整轨迹重建和最终贡献审查尚未执行。\n',
        '结构化记录：results/feedback_paused_isaac_20261007/partial_results_20261008.json、failure_status_20261008.json。冻结18文件与世界模型11文件哈希仍全部一致；服务器无新试验进程，未触碰现有渲染器。下一步先区分seed44候选非有限/超限原因及重定向拒绝、坐标桥和续接误差，再决定新版本验证，不能直接放宽安全门取得完整分数。\n'])
    with (ROOT/'reports/28_execution_feedback_paused_validation_20261007.md').open('a',encoding='utf-8') as stream:
        stream.write('\n'.join(section))
    print(json.dumps({'observed_beijing':date,'world_completed':13,'feedback_completed_seeds':[42,43],
                      'feedback_estimates':estimates},ensure_ascii=True))


if __name__ == '__main__':
    main()

# 文档导航

先按阅读目的选择入口；原始逐次尝试文档按日期保留，编号更大不自动代表效果更好。

## 首先阅读

- [仓库首页](../README.md) · [论文复现索引](../baselines/README.md)
- [PhyGile 独立索引](../baselines/phygile/INDEX.md) · [ReactiveBFM 独立索引](../baselines/reactivebfm/INDEX.md)
- [原基线结果汇总](results.md) · [分支与 PR 状态](branches_and_prs.md)
- [目录与记录约定](repository_map.md)

## 既往基线与执行

| 主题 | 入口 |
| --- | --- |
| DreamControl | [复现报告](../reports/02_dreamcontrol_reproduction.md) · [环境](../configs/env-dreamcontrol-r3.json) |
| CLoSD | [基线执行说明](closd_baseline_20261010.md) · [复现报告](../results/closd_baseline_20261010/README.md) |
| MuJoCo 跨仿真器验证 | [执行说明](mujoco_sim2sim_execution.md) · [结果报告](../reports/14_Completed_tasks_MuJoCo_sim2sim_20261002.md) |
| 训练可视化与记录 | [操作约定](training_visualization.md) |

## 执行反馈与测量诊断

- [物理引导实验协议](physics_guided_20261011.md)
- [重置与时刻对齐校验](reset_alignment_measurement_v3_20261011.md)
- [后续验证决策](remaining_validation_decision_20261012.md)
- [参考侧诊断审计](../results/support_reference_probe_v63_20261012/POST_RUN_AUDIT.md)
- [世界模型与执行反馈的历史可行性记录](../reports/21_feedback_world_model_feasibility_20261006.md)

这些是工程与研究诊断的历史记录。测量检查、接口连通或合成夹具通过，不等于生成质量、控制成功率或论文创新已经验证。

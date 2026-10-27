# 既往实验结果导航

根目录 `results/` 保留既往基线和执行反馈诊断的指标、回执及审计。新两篇论文的结果入口为：

- [PhyGile 原模块结果](../baselines/phygile/results/) · [Unitree 扩展结果](../baselines/phygile_unitree/results/v3/)
- [ReactiveBFM 单篇结果](../baselines/reactivebfm/results/) · [后续论文对齐档案](../baselines/paper_faithful_reproduction/README.md)

## 既往结果

| 主题 | 入口 |
| --- | --- |
| CLoSD 基线报告 | [closd_baseline_20261010](closd_baseline_20261010/README.md) |
| 已完成任务的跨仿真器评价 | [sim2sim_completed_20261002](sim2sim_completed_20261002/) |
| 人体动作生成质量 | [human_generation_quality_20261002](human_generation_quality_20261002/README.md) |
| 物理引导候选与对照 | [physics_guided_20261011](physics_guided_20261011/README.md) |
| 后续参考侧诊断 | [support_reference_probe_v63_20261012 审计](support_reference_probe_v63_20261012/POST_RUN_AUDIT.md) |

具体指标应沿各结果目录回溯配置、检查点和完整回合。测量校验、合成夹具及工程连通性不等于动作质量或算法收益；失败和提前终止必须与误差一并阅读。

# 物理引导条件动作生成：诊断实验记录

本目录保留用户指定方案的实现、对比、消融、失败与版本记录。它不代表已经验证的论文创新。

**直接阅读：[结果汇总与对比表](RESULTS.md) · [公开文件索引及散列](artifact_index.csv)。**

**重要有效性限制**：独立复核发现，旧轮参考与目标的创建坐标没有随实际角色的重置同步。不同种子和名义目标偏移因此混入初始化变化。旧轮数值仅作调试诊断，不能证明候选机制有效或无效。

- `experiment_report.html`：可在本地浏览器打开的完整报告；GitHub 文件页不会执行 HTML。
- `summary.json`、`per_attempt.csv`：10 个配置各 432 次独立回放，失败保留在分母。
- `all_construction_and_replay_attempts.csv`：74 个调度批次的 10,512 次构建或独立回放记录。
- `complete_attempt_register.json`：正式、开发、见证、失败运行和版本修订的索引。
- `cross_seed_coordinate_checks.json`、`evidence_status.json`：测量有效性问题，与数值进入门槛分开。
- `failure_case_catalog.json`：区分前缀终止、生成段终止、条件失败及物理代理失败，不将所有终止都称为跌倒。
- `budget_ledger.json`、`resource_summary.json`：失败时间与保守预算计费。
- `publication_versions.json`、`lineage_register.json`：版本、输入输出散列及绑定范围。
- `overleaf_experiment_record.tex`：已有 Overleaf 项目的增量研究记录。

独立初始化修复保存在邻接的 `reset_alignment*` 目录，不与旧轮数值合并。是否继续 G1 和下游训练，由独立测量与人体进入门槛决定；未执行阶段不填入模拟成绩。

原始人体动作、模型权重、SMPL 资产及派生轨迹不随公开记录发布。足部滑移和语义分数均为明示代理，不是接触或人工语义真值；人体仿真也不是 G1 或实机实验。

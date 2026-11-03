# 复现结果索引

本页按方法、协议与分母汇总入口。训练日志、工程测试、原始参考和策略执行结果分别阅读；只有对应评估和审核完成的数字才作为结果。

## DreamControl / CLoSD

| 范围 | 可阅读结果 | 说明 |
| --- | --- | --- |
| DreamControl 11 项任务 | [多任务报告](../reports/03_dreamcontrol_multitask_baselines.md) · [既往汇总](results.md) | 官方参考、任务各自判据；包含 0% 等失败结果 |
| CLoSD 公开权重控制评价 | [评价记录](../reports/13_CLoSD_official_checkpoint_eval_20261001.md) | 公开权重重评，不能说成本项目自训结果 |
| CLoSD 文本动作 | [报告及视频](../results/closd_baseline_20261010/README.md) | 与多任务控制成功率使用不同数据和协议 |

## PhyGile / ReactiveBFM

- **PhyGile：** [原跟踪与 Unitree 结果](../baselines/phygile/INDEX.md#实验结果与范围)。原诊断完整片段 26/128；Unitree 扩展未观察到改善。完整论文性能未复现。
- **ReactiveBFM：** [早期失败、数值恢复与后续检查点](../baselines/reactivebfm/INDEX.md#实验结果与范围)。13,001 次更新为历史已保存计数，单提示单种子诊断不能替代正式基准。

## TextOp / SuperMimic

### 主要指标：同一批 564 条生成留出

判据为全参考中，30 身体根相对位置误差均值的最大值 **≤ 0.3 米**；不因高度或躯干倾斜提前终止，缺帧或非有限值按失败保留分母。包含 6 条仅两帧的参考，总 58,674 帧。

| 策略 | 达标数 / 总数 | 达标率 |
| --- | ---: | ---: |
| 新 Vibed，2048 环境，4000 次优化 | 194 / 564 | 34.40% |
| PPO 第 5000 轮 | 202 / 564 | 35.8% |
| SuperMimic 128 环境探索 | 176 / 564 | 31.2% |
| SuperMimic 4096 环境、第 5000 次矢量采样迭代 | 201 / 564 | 35.6% |

4096 结果已独立重算全部轨迹、身体 FK 与指标，审核记录摘要见[验证与比较摘要](../baselines/textop/experiments/supermimic_vibed_20261103/results/validation_summary.json)。审核没有重新推导策略动作或重放全部物理。历史推理硬件、训练交互数和更新预算不同；根相对达标也不代表未倒地。

新 Vibed 的[外部评估与独立审核](../baselines/textop/experiments/supermimic_vibed_20261103/results/external_evaluation.md)覆盖全部 564 条和 58,674 帧，三个生成种子为 64/188、64/188、66/188，6 条两帧全部保留。较旧 4096 环境少 7 条、低约 1.24 个百分点；较 PPO 少 8 条、低约 1.42 个百分点，未显示超过这两项历史结果。共同模型、PD、零初速度及准确首末帧时序一致；训练预算、原生引擎与推理环境仍有差异，不证明统计优势或同质量效率。参考属于生成代理数据与纯仿真，非真实动作真值、论文 SnapMoGen 或实机结果。

### 分开的辅助协议

| 协议 | 已审核事实 / 状态 |
| --- | --- |
| 564 条、14 身体统一跟踪回报 | PPO 0.189114，4096 SuperMimic 0.170887；不是 0.3 米达标率 |
| IsaacLab 生成短参考，1880 次试验 | PPO5000 完整到参考末尾 1866 / 1880，14 次提前终止；不是逐帧精度验收 |
| 16 条原验证集长舞蹈，4096 SuperMimic | 在带物理提前终止的辅助协议下 0 / 16 到末尾；与 564 生成留出分开 |
| SnapMoGen | 重定向与帧边界有独立记录；完整策略测试成绩尚未完成 |
| SuperMimic 10000 后继统一回报 | 尚无已审核结果 |
| 新版 SuperMimic Vibed | 2026-11-03 主训练 4000 次优化 / 20,611,072 次交互完成，三个检查点与 160 行指标独立审核通过；全部 564 条外部评估与主指标审核已完成，194/564；这不是上述 14 身体回报 |

[新版代码和验证范围](../baselines/textop/experiments/supermimic_vibed_20261103/README.md) · [TextOp 总入口](../baselines/textop/INDEX.md) · [历史附件及报告清单](../baselines/textop/server_snapshot_20261102/README.md)

## 使用边界

新引擎、机器人碰撞体、控制频率、初始速度和策略自身观测格式均可能改变结果。比较时分别给出这些条件及训练成本；未完成的评价不填入成功率表。所有失败、缺失和短片段继续保留各自分母。

[总索引](project_index.md) · [报告导航](../reports/README.md) · [数据与附件](datasets_and_artifacts.md)

本轮 Vibed 视频：[全程数值筛查及限制](../baselines/textop/experiments/supermimic_vibed_20261103/results/external_evaluation.md#本轮长动作视频核查)。本批最长仅约4秒；19条至少100帧的达标候选均有明显根高度偏离参考，目前没有可展示的成功长动作视频。此结论不改变194/564根相对主指标。

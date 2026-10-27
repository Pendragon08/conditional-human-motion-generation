# 论文复现索引

本页区分既往基线与本次纳入主分支的两篇论文。各方法采用自己的代码、数据、配置和评价协议。

## 本次独立归档

| 论文 | 首先阅读 | 代码与配置 | 实验结果 | 报告 |
| --- | --- | --- | --- | --- |
| PhyGile | [归档索引](phygile/INDEX.md) | [scripts](phygile/scripts/) · [configs](phygile/configs/) · [独立实现组件](phygile/reimplementation/) | [原跟踪结果](phygile/results/) · [Unitree 扩展结果](phygile_unitree/results/v3/) | [原复现报告](phygile/reports/) · [Unitree 最终报告](phygile_unitree/reports/PhyGile_Unitree续训500次与闭环迁移最终报告_20261022.md) |
| ReactiveBFM | [归档索引](reactivebfm/INDEX.md) | [scripts](reactivebfm/scripts/) · [configs](reactivebfm/configs/) · [固定上游](reactivebfm/upstream/) | [训练与恢复结果](reactivebfm/results/) | [单篇报告](reactivebfm/reports/) · [后续 13,001 次检查点诊断](paper_faithful_reproduction/reports/heartbeat_20261023_2246.md) |

`phygile_unitree/` 是 PhyGile 的框架适配与跨仿真扩展，不另算一篇论文。`paper_faithful_reproduction/` 是这两篇论文后续联合核对、恢复和操作记录的历史档案，不替代各自的方法目录。

最新已归档操作见 [2026-10-24 暂停记录](paper_faithful_reproduction/reports/pause_cleanup_20261024.md)。本次只归档已有成果，不新增训练、评价或性能结论。

## 既往基线

| 方法 | 代码 / 配置说明 | 结果与报告 |
| --- | --- | --- |
| DreamControl | [复现说明](../reports/02_dreamcontrol_reproduction.md) · [环境配置](../configs/env-dreamcontrol-r3.json) | [11 项任务记录](../reports/03_dreamcontrol_multitask_baselines.md) |
| CLoSD | [执行说明](../docs/closd_baseline_20261010.md) · [环境配置](../configs/env-closd-baseline-20261010.json) | [报告入口](../results/closd_baseline_20261010/README.md) · [官方权重评估](../reports/13_CLoSD_official_checkpoint_eval_20261001.md) |

既往实现和证据继续放在仓库根目录的 `scripts/`、`configs/`、`results/`、`reports/`。新两篇的内容放在本目录各自子目录，避免混入旧实验。

## 阅读与复跑

1. 先看方法索引及报告中的范围、失败和缺失项。
2. 再核对配置、上游版本、输入来源、检查点与评价协议。
3. 需要附件时，再从对应 Release 获取指定文件；本页不自动下载模型或数据。
4. 复跑使用新的输出目录，不覆盖归档结果。

[返回首页](../README.md) · [分支与 PR 状态](../docs/branches_and_prs.md)

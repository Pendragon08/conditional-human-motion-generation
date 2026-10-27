# ReactiveBFM 复现归档索引

**归档范围：部分训练、数值恢复、检查点核验与 MuJoCo 闭环诊断。完整训练和论文基准尚未完成。**

本索引于 2026-10-27 整理。[原 README](README.md) 是较早交付快照，不是实时训练状态；后续记录保存在两篇论文的联合归档中。最后已归档操作见 [2026-10-24 暂停记录](../paper_faithful_reproduction/reports/pause_cleanup_20261024.md)。本次归档不启动训练。

## 代码、配置与环境

| 内容 | 入口 |
| --- | --- |
| 固定上游实现 | [upstream](upstream/) · [来源清单](upstream_manifest.json)，版本 `5a79d03cab824eb5917271ba720294e8a1b9d374` |
| 单篇复现工具与配置 | [scripts](scripts/) · [configs](configs/) |
| 环境与操作约定 | [环境](docs/environment.md) · [训练恢复约定](docs/full_run_operations.md) |
| 后续恢复代码 | [独立 recovery_v4](../paper_faithful_reproduction/recovery_v4/)；与先前尝试分别保留 |
| 后续异步仿真兼容层 | [执行脚本](../paper_faithful_reproduction/scripts/run_reactive_async.py) |

## 实验结果与范围

| 阶段 | 证据入口 | 已归档结论 |
| --- | --- | --- |
| 早期训练与诊断 | [原复现报告](reports/ReactiveBFM复现实验报告.md) · [full_training_20261021](results/full_training_20261021/) | 早期九个有界诊断均出现跌倒；失败记录保留 |
| FP32 数值恢复 v3 | [修复报告](reports/ReactiveBFM训练修复与PhyGile进程处理报告_20261022.md) · [恢复结果](results/recovery_fp32_v3/) | 有限值、优化器计数与恢复一致性核查；不代表长训练已稳定或论文性能已达到 |
| 后续 13,001 次更新检查点 | [检查点与闭环报告](../paper_faithful_reproduction/reports/heartbeat_20261023_2246.md) | 模型、EMA、优化器已留档核验；一次 `walk forward` / seed 42 诊断完成 500 个控制步、20 次重规划，0/500 帧越过该诊断阈值 |
| 最后已归档操作状态 | [2026-10-24 记录](../paper_faithful_reproduction/reports/pause_cleanup_20261024.md) | 复现与自动推进暂停，保留 13,001 次更新恢复点；不是“完整训练完成” |

13,001 是已保存、可核验检查点的实际更新计数，不应当作论文完整训练的完成量。单提示、单种子的近似诊断通过，也不等于正式测试集成功率、实时推理或实机性能。

## 报告与审计

- [单篇报告目录](reports/) · [单篇结果目录](results/)
- [论文对齐综合进展](../paper_faithful_reproduction/reports/reproduction_progress_20261023.md)
- [FP32 v3 审计](results/recovery_fp32_v3/final_independent_audit.md)：保留 provisional WARN、后续待验证项及原始失败
- [数据保留说明](docs/data_preservation.md)

## 检查点与附件

- [原完整训练阶段附件](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/reactivebfm-full-training-20261022)
- [FP32 恢复 v3 附件](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/reactivebfm-recovery-fp32-v3-20261022)
- [后续 13,001 检查点及进展附件](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/paper-reproduction-progress-20261023)

大文件按需下载，并按对应记录核对大小与哈希。上游代码、外部数据、人体模型及预训练跟踪器分别遵循自己的许可。

## 尚未完成

完整 1M 更新目标、未公开的目标 / reaching 数据、完整论文评价、实时 / TensorRT 性能和实机验证仍不能由当前诊断替代。

[论文复现索引](../README.md) · [仓库首页](../../README.md)

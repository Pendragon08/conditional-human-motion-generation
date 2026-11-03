# 仓库总索引

更新日期：2026-11-03。按“代码、数据、复现结果、报告、大文件”进入，历史记录按原日期阅读。

| 项目 | 代码与配置 | 数据与大文件 | 复现结果 / 报告 |
| --- | --- | --- | --- |
| DreamControl | [代码入口](code_index.md#dreamcontrol--closd) | [数据与附件](datasets_and_artifacts.md) | [11 项任务](../reports/03_dreamcontrol_multitask_baselines.md) |
| CLoSD | [执行说明](closd_baseline_20261010.md) · [运行脚本](../scripts/) | [动作数据与权重说明](datasets_and_artifacts.md) | [文本动作报告](../results/closd_baseline_20261010/README.md) · [官方权重评价](../reports/13_CLoSD_official_checkpoint_eval_20261001.md) |
| PhyGile | [单篇目录](../baselines/phygile/INDEX.md) · [Unitree 扩展](../baselines/phygile_unitree/) | [AMASS 来源](../baselines/phygile/datasets/AMASS/README.md) | [结果与范围](../baselines/phygile/INDEX.md#实验结果与范围) |
| ReactiveBFM | [代码和环境](../baselines/reactivebfm/INDEX.md#代码配置与环境) | [数据保留记录](../baselines/reactivebfm/docs/data_preservation.md) | [结果与限制](../baselines/reactivebfm/INDEX.md#实验结果与范围) |
| TextOp / SuperMimic | [项目索引](../baselines/textop/INDEX.md) · [11 月 3 日适配代码](../baselines/textop/experiments/supermimic_vibed_20261103/README.md) | [数据来源与草稿范围](datasets_and_artifacts.md) | [统一结果入口](result_index.md#textop--supermimic) · [历史代码与附件快照](../baselines/textop/server_snapshot_20261102/README.md) |

## 按用途查找

- **找代码：** [代码与环境索引](code_index.md)，区分固定上游、运行适配和历史工具。
- **找数据：** [数据与附件索引](datasets_and_artifacts.md)，区分训练参考、生成留出、策略轨迹和第三方来源。
- **看结论：** [复现结果索引](result_index.md)，明确判据、分母、失败和未完成项。
- **找文件下载：** [Releases](https://github.com/Pendragon08/conditional-human-motion-generation/releases) · [机器可读公开附件清单](release_inventory.json)。
- **看完整报告：** [报告导航](../reports/README.md)；TextOp 报告及大文件的草稿状态另见其项目索引。

## 文件状态

| 标记 | 含义 |
| --- | --- |
| Git 已归档 | 当前分支可阅读的代码、配置、小型结果或报告 |
| 公开 Release | 已发布附件；使用具体文件及 SHA-256 获取 |
| Release 草稿 | 尚未发布，仅有权限的账户可查看；不等于公开下载 |
| 本地 / 服务器记录 | 原始运行证据或运行中工件；不能假定已全部同步到 GitHub |
| 历史快照 | 固定日期记录；其中的 running / CURRENT 不是实时状态 |

整理前主分支有 11,151 个 Git 文件。目录与历史相对路径继续有效；最新的分类入口与机器清单指向各自原文件。大型模型、原始动作与外部软件仍按各自许可和可用性获取。

[首页](../README.md) · [目录约定](repository_map.md) · [机器可读项目目录](reproduction_catalog.json)

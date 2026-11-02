# 数据集、结果与附件索引

本页按用途索引仓库，不移动或重命名历史文件。公开代码、历史回执与可下载数据是不同范围；每项实验的来源、切分、许可、版本和校验值以其清单及报告为准。

## 按方法查找

| 方法 / 数据用途 | 数据、来源与配置 | 结果和报告 | 获取状态 |
| --- | --- | --- | --- |
| DreamControl / CLoSD | [根目录配置](../configs/) · [运行脚本](../scripts/) | [结果导航](../results/README.md) · [报告导航](../reports/README.md) | 既往已归档内容按各报告与对应 Release 获取 |
| PhyGile | [AMASS 数据说明](../baselines/phygile/datasets/AMASS/README.md) · [来源清单](../baselines/phygile/datasets/AMASS/assets.json) | [PhyGile 索引](../baselines/phygile/INDEX.md) · [Unitree 扩展](../baselines/phygile_unitree/) | 数据与人体模型须核各自来源和使用条件 |
| ReactiveBFM | [环境与数据保留说明](../baselines/reactivebfm/docs/) · [配置](../baselines/reactivebfm/configs/) | [结果及报告索引](../baselines/reactivebfm/INDEX.md) | 已归档结果的范围以方法索引及对应附件为准 |
| TextOp / SuperMimic | [快照源码和配置](../baselines/textop/server_snapshot_20261102/) · [数据准备记录](../baselines/textop/reports/dataset_preparation_20261025.md) | [TextOp 索引](../baselines/textop/INDEX.md) · [语义汇总](../baselines/textop/results/semantic_evaluator_20261029/tmr100_semantic32_v1/aggregate_public.json) | 大型训练参考库、检查点和结果包部分位于尚未发布的 Release 草稿 |

## TextOp 数据与结果如何区分

1. **原始训练参考与生成参考**分别归档，不能混作一个数据集；参考轨迹并非策略执行轨迹。训练输入记录中原始与生成参考共 8077 段，其范围与归档文件数口径不同。
2. **生成留出评估**、**原始验证集**和 **SnapMoGen 测试集**为不同来源与协议；分别保留种子、片段、失败和分母。生成留出 564 条的统一回报不是 SnapMoGen 全集成绩。
3. **代码仓库中的小型配置、清单、报告与结构化结果**可从各目录阅读；大型模型、数据分片和轨迹包的名称及 SHA-256 见[服务器快照与附件索引](../baselines/textop/server_snapshot_20261102/README.md)。索引列出草稿并不使草稿公开可下载。
4. **HTML 报告**是按对应日期冻结的版本。最新本地报告和草稿中已有版本可能不一致；以附件名称和摘要区分，不能假定最新版已公开。

## 使用与许可

- Git 大文件指针、Release 附件和第三方下载源是不同存储位置；使用前按方法索引确认文件是否实际可访问、是否完整，以及对应 SHA-256。
- AMASS、人体模型、其他动作数据、上游模型与软件各有独立权属及使用条款。用户提供的授权应按具体材料及署名条件核对，不推定覆盖其他模型或软件。
- 历史结果和失败记录继续保留；新的运行使用独立输出目录，记录源版本、输入、检查点、仿真器、种子和评价协议。

[返回仓库首页](../README.md) · [目录约定](repository_map.md) · [论文复现索引](../baselines/README.md)

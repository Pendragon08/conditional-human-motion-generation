# 数据集、结果与附件索引

本页按用途索引仓库，不移动或重命名历史文件。公开代码、历史回执与可下载数据是不同范围；每项实验的来源、切分、许可、版本和校验值以其清单及报告为准。

## 按文件类型查找

| 类型 | 入口 / 可用性 |
| --- | --- |
| 固定源码与适配 | [代码索引](code_index.md) · [新版 50 Hz NPZ 适配](../baselines/textop/experiments/supermimic_vibed_20261103/README.md) |
| 原始或第三方训练输入 | 按各方法来源清单及许可获取，Git 中的清单不等于数据文件 |
| G1 原始训练参考 / 增强参考 | 8077 段训练清单；大型 FK 分片见 TextOp 草稿附件索引 |
| 生成留出参考 | 564 段、58,674 帧、6 条两帧，独立于训练清单和原验证集 |
| 策略执行轨迹 / 指标 | [复现结果索引](result_index.md)，按控制协议与分母区分 |
| 已公开模型、视频和归档 | [公开附件机器清单](release_inventory.json)，记录具体大小与 GitHub SHA-256 |
| TextOp 未发布大型文件 | Release 草稿 431174357，当前 49 项；尚未公开，既有[附件索引](../baselines/textop/server_snapshot_20261102/README.md)仍按各冻结版本追溯 |

截至 2026-11-03，已只读核对 19 个公开 Release、198 项公开附件；本次核实草稿含 49 项附件。该清单核的是 GitHub 元数据，并非本次重新下载或逐项解包验证全部附件。已删除的历史 CLoSD / PhyGile 训练权重不属于完整仓库备份范围；仍存在的关键公开权重以实际附件为准。

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

## TextOp 上游训练数据来源（历史记录）

原训练资产的[固定版本清单](../baselines/textop/configs/asset_manifest.json)记录了上游 [Yochish/TextOp-Data 的 bb9604ce 版本](https://huggingface.co/datasets/Yochish/TextOp-Data/tree/bb9604ce36ad4c0a77b5cccde89ebbb7c3ca93c9)。其中 `BABEL-AMASS-ROBOT-23dof-FULL-50fps/train.pkl` 记录为 4,905,919,425 字节，来源清单的 LFS SHA-256 为 `33123a15905fb098b5fa6c083a1dc76441e7c2a90b190844534aeba1fc0dd331`。当时的[下载完成记录](../baselines/textop/results/download_complete.json)记载该文件的实际摘要与之匹配；这是历史下载证据，本次未重新读取整个文件，也不保证上游链接当前可用。

该上游训练输入与本项目后来生成的 G1 参考库、策略轨迹及实验结果应分别追溯；来源清单不表示它们已合并成同一份数据，也不代表全部材料可自由再分发。

## 使用与许可

- Git 大文件指针、Release 附件和第三方下载源是不同存储位置；使用前按方法索引确认文件是否实际可访问、是否完整，以及对应 SHA-256。
- AMASS、人体模型、其他动作数据、上游模型与软件各有独立权属及使用条款。用户提供的授权应按具体材料及署名条件核对，不推定覆盖其他模型或软件。
- 历史结果和失败记录继续保留；新的运行使用独立输出目录，记录源版本、输入、检查点、仿真器、种子和评价协议。

[返回仓库首页](../README.md) · [目录约定](repository_map.md) · [论文复现索引](../baselines/README.md)

## 2026-11-03 Vibed 最终模型备份

经用户明确授权，最终小检查点已直接从服务器备份至上述草稿，71,781,529 字节、asset ID 705833338。原文件与 GitHub 官方摘要一致，[完整记录](../baselines/textop/experiments/supermimic_vibed_20261103/results/checkpoint_backup.json)保留原客户端收尾错误及后续独立核验。模型负载未经过本机 VPN 中转，原件仍在服务器。草稿未发布；本次备份不包含完整优化器检查点、其他迭代或全量新轨迹。

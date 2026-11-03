# 目录与记录约定

## 代码与结果按方法归属

| 范围 | 代码 / 配置 | 结果 / 报告 |
| --- | --- | --- |
| 既往 DreamControl、CLoSD 与执行反馈诊断 | 根目录 `scripts/`、`configs/` | 根目录 `results/`、`reports/`、`media/` |
| PhyGile 原模块复现 | `baselines/phygile/` 内的 `upstream/`、`scripts/`、`reimplementation/`、`configs/` | 同目录 `results/`、`reports/` |
| PhyGile Unitree 扩展 | `baselines/phygile_unitree/scripts/`、`configs/` | 同目录 `results/`、`reports/` |
| ReactiveBFM | `baselines/reactivebfm/upstream/`、`scripts/`、`configs/` | 同目录 `results/`、`reports/` |
| TextOp / SuperMimic | `baselines/textop/scripts/`、`configs/`、`upstream/` 与 `server_snapshot_20261102/` | 同目录 `results/`、`reports/`；大文件见快照附件清单 |
| 两篇论文后续联合核对与恢复档案 | `baselines/paper_faithful_reproduction/` | 同目录保留按日期组织的历史证据 |

## 阅读顺序

从 [论文复现索引](../baselines/README.md) 进入单篇索引，再查看对应报告、结果和来源清单。既往文档、脚本和结果不因目录整理而移动，以免破坏运行路径、相对链接或历史来源校验。

## 证据与附件

- `reports/` 解释结果；`results/` 保存结构化指标、执行回执、失败和审计。
- `upstream/` 与来源清单标识固定上游；独立实现和运行适配单独记录。
- 审计和发布清单对应其标注的冻结提交，不能把后来新增的导航文件当成原快照的一部分。
- 检查点、视频和恢复包以 Release 为附件入口，按需获取；历史附件不因本次整理重新上传。
- 新运行使用新的输出目录，并记录输入、源版本、配置、种子和评价协议，避免覆盖旧证据。
- 受限原始数据与人体模型按许可另行获取，不能把代码公开归档解释为原始资产可以再分发。

[原 README 固定快照](https://github.com/Pendragon08/conditional-human-motion-generation/blob/9f9a12760c1b9ce2146a48fa387aabcafde8168e/README.md) · [分支与 PR](branches_and_prs.md)

数据来源、结果与大文件状态见 [数据与附件索引](datasets_and_artifacts.md)。

## 2026-11-03 分类入口

- [仓库总索引](project_index.md)、[代码](code_index.md)、[复现结果](result_index.md)、[公开附件清单](release_inventory.json) 分别承接导航、运行源码、已审核结论和下载元数据。
- 新的 SuperMimic Vibed 导出位于 [独立实验目录](../baselines/textop/experiments/supermimic_vibed_20261103/README.md)，与 11 月 2 日冻结快照分别保存。导出清单明确便携路径改动，原始快照摘要继续有效。
- 新导航与摘要不会使历史 Release 草稿公开，也不表示服务器所有文件或已删除权重都有完整备份。

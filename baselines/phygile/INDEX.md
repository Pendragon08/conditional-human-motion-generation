# PhyGile 复现归档索引

**归档范围：部分复现、G1 跟踪、独立实现的生成组件及 Unitree 框架扩展。完整论文性能尚未复现。**

本索引于 2026-10-27 整理。原 README、报告、源码快照和结果文件保留原样，其“正在运行 / 下载”等描述只代表各自记录日期。最后已归档操作见 [2026-10-24 暂停记录](../paper_faithful_reproduction/reports/pause_cleanup_20261024.md)。

## 代码、配置与环境

| 内容 | 入口 |
| --- | --- |
| 原公开跟踪代码及版本 | [upstream](upstream/) · [来源清单](upstream_manifest.json) |
| 本次运行与评价工具 | [scripts](scripts/) · [configs](configs/) |
| 缺失生成组件的独立实现 | [reimplementation](reimplementation/README.md)；不能视为作者提供的完整生成器 |
| 原实验环境与复现范围 | [环境](docs/environment.md) · [范围](docs/reproduction_scope.md) |
| Unitree 框架适配 | [单独目录](../phygile_unitree/README.md) · [脚本](../phygile_unitree/scripts/) · [配置](../phygile_unitree/configs/) |

## 实验结果与范围

下表两行使用不同数据和协议，不能合并计算整体成功率。

| 实验 | 结果入口 | 已归档结论 |
| --- | --- | --- |
| 原跟踪器累计 2,000 轮 PPO | [训练与生成组件报告](reports/PhyGile续训与生成核心报告.md) · [原始结果](results/continuation_2000/) | 固定诊断协议下测试完整片段 26/128；一个来源组、一个训练种子，不是论文主表复现 |
| Unitree 适配后的 500 轮新增 PPO | [最终报告](../phygile_unitree/reports/PhyGile_Unitree续训500次与闭环迁移最终报告_20261022.md) · [结构化结果](../phygile_unitree/results/v3/summary.json) | 四段开发诊断中，Isaac 完成数 2/4→1/4，MuJoCo 为 1/4→1/4；未观察到改善，保留父策略作为该诊断基线 |

262D 接口、扩散采样与仿真闭环的工程检查，不等于训练好的文本条件生成能力。未训练生成器的真实候选未获得有效提交，不能报告为稳定动作生成或完整闭环性能。

## 报告与证据

- [课程评估与 262D 闭环补全](reports/PhyGile课程评估与262D闭环补全报告.md)
- [HumanML3D 到 G1 配对数据处理](reports/HumanML3D到G1配对数据处理报告.md)
- [Unitree V3 证据索引](../phygile_unitree/results/v3/evidence_index.json)
- [Unitree V3 独立审计](../phygile_unitree/EXPERIMENT_AUDIT_V3.md)：保留 WARN / provisional 及失败记录
- [原始完整说明](README.md)：历史交付快照

## 检查点与附件

- [2,000 轮跟踪实验 Release](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/phygile-tracking-2000updates-20261017)
- [Unitree 续训与跨仿真 Release](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/phygile-unitree-headless-20261022)
- [后续论文对齐进展归档](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/paper-reproduction-progress-20261023)

附件按需下载。完整训练检查点、仅供推理的资产与原始数据分别遵循自己的恢复能力及许可要求。

## 尚未完成

完整可用数据范围、论文课程与难度标签、充分训练的条件生成器、生成分布上的控制适配和正式全量评价仍不能由当前小规模结果替代。没有本项目实机部署结果。

[论文复现索引](../README.md) · [仓库首页](../../README.md)

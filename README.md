# Whole-Body Humanoid Control with Conditional Motion Priors

<div align="center">

**条件动作先验、物理执行与全身人形机器人控制**

[论文复现索引](baselines/README.md) · [实验结果](docs/results.md) · [报告导航](reports/README.md) · [文档导航](docs/README.md) · [分支与 PR](docs/branches_and_prs.md)

</div>

本仓库保存全身人形机器人控制相关的基线复现、参考动作生成与精化、物理跟踪、执行反馈及跨仿真器诊断。这里是复现与研究记录，不是所列论文的官方实现；每项结果均需结合自己的配置、检查点和评价协议阅读。

## Reproductions

| 论文 / 方法 | 主要内容 | 入口与状态 |
| --- | --- | --- |
| DreamControl | 官方参考动作下的 11 项控制任务、训练与任务评价 | [既往复现报告](reports/02_dreamcontrol_reproduction.md) · [多任务结果](reports/03_dreamcontrol_multitask_baselines.md) |
| CLoSD | 已发布权重评价、文本动作报告、训练尝试与失败分析 | [复现报告](results/closd_baseline_20261010/README.md) · [执行说明](docs/closd_baseline_20261010.md) |
| PhyGile | 跟踪模块、生成组件独立实现、G1 适配与跨仿真诊断 | **[独立归档入口](baselines/phygile/INDEX.md)**；部分复现，完整论文性能未复现 |
| ReactiveBFM | 有界训练、数值恢复、检查点核验及 MuJoCo 闭环诊断 | **[独立归档入口](baselines/reactivebfm/INDEX.md)**；部分复现，完整训练与论文基准未完成 |

PhyGile 与 ReactiveBFM 的代码、配置、结果和报告分别放在 `baselines/` 下各自的目录。以往 DreamControl / CLoSD 的 `scripts/`、`configs/`、`results/`、`reports/` 路径继续作为原归档入口。

截至本次整理，PhyGile / ReactiveBFM 最后已归档的操作状态为 [2026-10-24 暂停记录](baselines/paper_faithful_reproduction/reports/pause_cleanup_20261024.md)。历史日志中的“running”仅表示记录当时的状态。归档合并不表示论文性能验证通过，也不恢复训练。

## Setup

只阅读结果时，直接使用上面的在线入口。需要代码时，可按论文选择目录，避免默认取得全部历史文件和 LFS 资产。以下为 Bash 示例：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --sparse https://github.com/Pendragon08/conditional-human-motion-generation.git
cd conditional-human-motion-generation
git lfs install --local --skip-smudge
# 示例：只展开 PhyGile 及其 Unitree 扩展
git sparse-checkout set baselines/phygile baselines/phygile_unitree
```

不同基线使用不同环境。先查看 [论文复现索引](baselines/README.md) 和各方法的环境说明。SMPL / SMPL-X、AMASS、HumanML3D、机器人资产及上游权重需按各自条款获取。检查点、视频和离线包在 [Releases](https://github.com/Pendragon08/conditional-human-motion-generation/releases) 中按需下载。

## Workflow

```text
文本 / 目标条件 + 人体动作先验
              ↓
         条件参考动作
              ↓
   重定向 / 物理精化（按方法选择）
              ↓
      动作跟踪与控制策略学习
              ↓
      任务评价 / 执行反馈诊断
```

这是研究流程图，各论文的实际实现和证据范围分别记录，不能把不同基线拼接成已经验证的统一算法。

## Results

- **既往基线：** [DreamControl / CLoSD 原始评价汇总](docs/results.md)，保留原协议、数值和失败说明。
- **PhyGile：** [独立结果入口](baselines/phygile/INDEX.md#实验结果与范围)，区分原跟踪实验与 Unitree 扩展。
- **ReactiveBFM：** [独立结果入口](baselines/reactivebfm/INDEX.md#实验结果与范围)，区分早期失败、恢复检查点和单回合诊断。
- **候选方法：** [后续验证决策](docs/remaining_validation_decision_20261012.md)；现有证据不足以支持可靠的下游收益。

本仓库没有自己的真实机器人部署结论。模拟人体角色、G1 动力学仿真和跨仿真器测试分别标注，不能相互替代。

## Repository Structure

```text
baselines/
  phygile/                       PhyGile 代码、配置、结果与报告
  phygile_unitree/                PhyGile 的 G1 / Unitree 框架扩展
  reactivebfm/                    ReactiveBFM 代码、配置、结果与报告
  paper_faithful_reproduction/    两篇论文的联合核对、恢复和历史操作档案
configs/  scripts/                既往基线的配置与运行工具
results/  reports/                既往结果证据与可阅读报告
docs/                            总导航、方法说明和测量诊断
media/  checkpoints/              既往展示页与部分 LFS 检查点
plans/  .aris/                    历史研究计划与执行记录
```

完整说明见 [目录与记录约定](docs/repository_map.md)。保留历史路径和文件内容，使旧报告、结果清单及来源校验仍可追溯。

## Documentation

- [文档导航](docs/README.md) · [报告导航](reports/README.md) · [结果导航](results/README.md)
- [综合实验汇报](reports/teacher_report_20261002/README.md) · [演示视频目录](media/DreamControl_official_baselines.html)
- [分支与 PR 状态](docs/branches_and_prs.md) · [机器可读复现目录](docs/reproduction_catalog.json)
- [整理前 README 的固定版本](https://github.com/Pendragon08/conditional-human-motion-generation/blob/9f9a12760c1b9ce2146a48fa387aabcafde8168e/README.md)

## Acknowledgements and License

感谢 [DreamControl](https://github.com/GenRobo/DreamControl)、[CLoSD](https://github.com/GuyTevet/CLoSD)、[PhyGile tracking](https://github.com/Baojch/phygile_tracking) 和 [ReactiveBFM](https://github.com/zjwzcx/ReactiveBFM) 的公开实现。固定版本、改动范围和引用要求见各方法目录。

上游代码、数据、人体模型、机器人资产及检查点遵循各自许可。仓库没有统一的顶层许可证；公开归档不改变上游条款。受限原始数据与模型不通过本次整理公开发布。

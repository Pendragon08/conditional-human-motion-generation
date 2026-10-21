# Whole-Body Humanoid Control with Conditional Motion Priors

<div align="center">

**条件动作先验、物理执行与全身人形机器人控制**

[实验汇报](reports/teacher_report_20261002/README.md) · [CLoSD 复现报告](results/closd_baseline_20261010/README.md) · [基线结果](#results) · [演示视频](media/DreamControl_official_baselines.html) · [复现实验](#setup)

</div>

本仓库研究如何利用条件生成模型与人体动作先验，支持人形机器人的全身技能控制。保存的内容包括 **DreamControl / CLoSD 基线复现、参考动作生成与精化、执行反馈诊断，以及 Isaac → MuJoCo 跨仿真器评价**。生成动作是控制流程的输入之一，最终需通过实际仿真执行和任务评价检验其作用。

```text
文本 / 目标条件 + 人体动作先验
              ↓
         条件参考动作
              ↓
  物理精化 / 重定向（按基线流程）
              ↓
       动作跟踪与策略学习
              ↓
     任务评价 / 执行反馈分析
```

## Project Status

- [x] DreamControl 的 11 项控制任务均已保存训练与官方评估结果。
- [x] CLoSD 已发布权重的多任务评估完成；训练复现与权重评估分别记录。
- [x] 保存部分任务的 MuJoCo 迁移结果、失败案例和测量校验。
- [ ] 候选执行反馈方法尚无充分证据支持联合质量或下游学习收益。
- [ ] 没有本项目的真实机器人部署结果。

这是复现与研究仓库，不是上述论文的官方代码。下文汇总已归档证据，不表示实验正在运行。后续创新验证按用户要求暂停，旧日志中的运行状态仅反映记录当时的情况。

## Table of Contents

1. [Setup](#setup)
2. [Phase 1: Reference Motion Generation](#phase-1-reference-motion-generation)
3. [Phase 2: Physics-Based Control](#phase-2-physics-based-control)
4. [Phase 3: Evaluation and Feedback](#phase-3-evaluation-and-feedback)
5. [Results](#results)
6. [Repository Structure](#repository-structure)
7. [Documentation](#documentation)
8. [Acknowledgements and License](#acknowledgements-and-license)

## Setup

先获取代码和报告，再按需要下载 LFS 文件：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Pendragon08/conditional-human-motion-generation.git
cd conditional-human-motion-generation
git lfs install
# 可选：只下载仓库中已发布的检查点
git lfs pull --include="checkpoints/*"
```

两个基线使用不同的运行环境，应分别配置。

| 流程 | 已记录环境 | 配置与复现记录 |
|---|---|---|
| DreamControl / G1 | Isaac Sim 4.5.0、上游附带的 Isaac Lab 训练工程、Python 3.10、PyTorch 2.5.1 | [环境配置](configs/env-dreamcontrol-r3.json) · [复现记录](reports/02_dreamcontrol_reproduction.md) |
| CLoSD / SMPL 模拟角色 | Isaac Gym Preview 4、Python 3.8.20、PyTorch 1.13.1 | [环境配置](configs/env-closd-r3.json) · [基线说明](docs/closd_baseline_20261010.md) |
| 跨仿真器评价 | MuJoCo、冻结控制策略 | [环境配置](configs/mujoco_sim2sim_env.json) · [执行说明](docs/mujoco_sim2sim_execution.md) |

上游代码版本分别固定为 DreamControl `ec28b508313d6837de52c68247fbc8cf81fe4935` 和 CLoSD `d8950bdb78312fcd2844f336b3fc079e7368c87f`。安装入口分别参照 [DreamControl](https://github.com/GenRobo/DreamControl) 与 [CLoSD](https://github.com/GuyTevet/CLoSD)，再对照本仓库的环境记录。

本仓库不是包含全部依赖的一键安装包。部分脚本沿用原服务器的解释器、缓存和资产路径，迁移前需要逐项适配。SMPL、AMASS、HumanML3D 和上游模型需按各自条款另行获取；本仓库中的少量控制检查点不能替代这些依赖。

## Phase 1: Reference Motion Generation

**DreamControl 路线：** 条件人体动作生成 → G1 重定向 → 轨迹修正 → 控制训练。生成、重定向和精化分别有独立入口：

| 环节 | 实现入口 |
|---|---|
| 条件动作生成 | [r3_dreamcontrol_generate_safe.sh](scripts/r3_dreamcontrol_generate_safe.sh) |
| 动作重定向 | [r3_dreamcontrol_retarget_safe.sh](scripts/r3_dreamcontrol_retarget_safe.sh) |
| 轨迹精化 | [r3_dreamcontrol_refine_safe.sh](scripts/r3_dreamcontrol_refine_safe.sh) |
| 生成与控制证据边界 | [复现记录](reports/02_dreamcontrol_reproduction.md) |

**CLoSD 路线：** DiP 根据文本、动作前缀和目标条件规划动作，物理跟踪器执行，并将执行状态反馈给后续规划。其模拟对象是 SMPL 人体角色，不能直接视为 G1 执行。固定版本、权重和录制流程见 [CLoSD 基线说明](docs/closd_baseline_20261010.md)。

官方参考库、重新生成的动作和执行后导出的轨迹分别保存来源，不混记为同一类训练数据。

## Phase 2: Physics-Based Control

DreamControl 的归档基线使用官方任务参考动作和 PPO 控制训练；CLoSD 的已发布权重评估与本地微调实验独立记录。

| 内容 | 入口 |
|---|---|
| DreamControl 多任务配置、训练及评估 | [多任务复现记录](reports/03_dreamcontrol_multitask_baselines.md) |
| CLoSD 官方权重评估 | [评估记录](reports/13_CLoSD_official_checkpoint_eval_20261001.md) |
| CLoSD 本地训练尝试与失败分析 | [训练记录](reports/01_closd_reproduction.md) |
| PPO 曲线、策略文件与导出说明 | [训练解读](reports/16_PPO_training_interpretation_and_policy_formats_20261002.md) |

启动新的正式 RL 训练前，按 [训练可视化说明](docs/training_visualization.md) 检查实际训练场景、输出目录和资源预算。独立检查点回放与历史视频不等于训练直播。

## Phase 3: Evaluation and Feedback

评价分为三个层次：

1. **参考动作质量：** 条件误差、接触代理、脚滑、穿透、连续性与多样性。
2. **执行表现：** 固定策略下的任务成功、跟踪误差、提前终止与稳定性。
3. **数据训练价值：** 相同控制框架与预算下，比较不同参考数据的下游收益。

条件纠偏、等预算候选筛选及相应消融见 [实验协议](docs/physics_guided_20261011.md)。早期测量不一致的结果保留为诊断记录；修复后的对照和后续参考侧诊断分别归档，不能合并成有效性结论。当前候选方法未满足继续 G1 与下游训练的条件，见 [后续验证决策](docs/remaining_validation_decision_20261012.md) 和 [参考侧诊断审计](results/support_reference_probe_v63_20261012/POST_RUN_AUDIT.md)。

## Results

### DreamControl: archived control baselines

使用官方参考动作，8192 个并行环境、2000 次训练迭代、seed 42；以官方评价器在 1000 环境、500 帧下评价。部分任务有明确记录的恢复或显示 I/O 适配，详见 [原始报告](reports/03_dreamcontrol_multitask_baselines.md)。

| 任务 | 成功率 | 任务 | 成功率 |
|---|---:|---|---:|
| Kick | 85.3% | Punch | 87.2% |
| Jump | 100.0% | Sit | 100.0% |
| Pick | 93.9% | Bimanual Pick | 99.9% |
| Ground Pick | 38.1% | Ground Pick Top | 0.0% |
| Button Press | 99.9% | Open Drawer | 100.0% |
| Open Door | 99.0% | | |

11 项均已完成评价，不等于 11 项均达到论文性能，也不代表生成方法创新已得到验证。成功率遵循官方任务判据，不能扩展解释为自然抓握、持续接触或未见场景泛化。

### CLoSD: text-to-motion baseline report

[完整报告与视频](results/closd_baseline_20261010/README.md) 已同步至 GitHub，提供可离线打开的 HTML 与下载包。报告包含作者公开动作重评、20,000 条新采集动作评测、9 段任务演示与 3 段生成动作预览。报告内容更新于 2026-10-17，补充发布于 2026-10-21；原指标不变，没有新增训练。该文本动作评测与下表多任务控制评价分别记录。

### CLoSD: released-checkpoint evaluation

4096 环境、500 帧、seed 0，每个状态至少 1000 次重置。来源：[评估记录](reports/13_CLoSD_official_checkpoint_eval_20261001.md)。

| 官方已发布权重 | REACH | STRIKE | SIT | GET_UP |
|---|---:|---:|---:|---:|
| 未微调 | 100% | 85% | 76% | 56% |
| 已微调 | 100% | 94% | 89% | 97% |

STRIKE 合并拳击与踢击计数。这是官方权重的评价复现，不是本项目重新训练取得的成绩。

### Cross-simulator and candidate-method findings

MuJoCo 测试已保存成功与失败案例，部分任务存在跌倒、接触或物体抬升失败；见 [跨仿真器结果](reports/14_Completed_tasks_MuJoCo_sim2sim_20261002.md) 与 [扰动和实体接触实验](reports/11_Kick_Punch_robustness_contact_20261001.md)。本项目尚未建立候选执行反馈机制的可靠下游收益，也没有实机测试结论。

## Repository Structure

```text
configs/       环境、任务与冻结实验配置
scripts/       生成、精化、控制、评价及导出工具
plans/         研究计划与实验方案
docs/          分阶段执行说明与测量诊断
results/       指标、逐次尝试记录、校验与审计
reports/       可阅读的结果分析与汇报
media/         评估视频、图表与离线展示页
checkpoints/   已发布的部分控制检查点（Git LFS）
```

## Documentation

- [综合实验汇报](reports/teacher_report_20261002/README.md)
- [DreamControl 复现问题](reports/18_DreamControl_reproduction_issues_20261006.md) · [CLoSD 复现问题](reports/19_CLoSD_reproduction_issues_20261006.md)
- [执行反馈与世界模型探索](reports/21_feedback_world_model_feasibility_20261006.md)
- [整理前的完整 README](https://github.com/Pendragon08/conditional-human-motion-generation/blob/ddcc1f47010a397b2e11f21d56ac43396bfa6b0b/README.md)

仓库内 HTML 页面可下载后用浏览器打开；GitHub 文件页不一定直接播放离线报告中的媒体。

## Acknowledgements and License

感谢 [DreamControl](https://github.com/GenRobo/DreamControl) 与 [CLoSD](https://github.com/GuyTevet/CLoSD) 提供的生成、控制和评价基础。上游方法与本仓库改动的界限、提交版本及运行适配均记录在对应复现报告中。

仓库目前未提供统一的顶层 LICENSE。上游代码、人体模型、数据、机器人资产和检查点遵循各自许可；公开仓库不改变这些条款。引用复现结果时请同时注明上游论文、本仓库提交版本与具体实验协议。

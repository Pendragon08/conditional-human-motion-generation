# Conditional Human Motion Generation

本仓库记录**条件驱动人体运动生成**课题的可复现实验。研究流程是条件输入 → 生成人体运动 → 物理／接触修正 → 机器人参考动作 → 固定下游执行与反馈。机器人跟踪用于检验生成动作是否可执行，并非本课题的唯一研究对象。本仓库与原人形机器人格斗项目分开。

2026-10-01：用户决定暂不继续创新点验证，先完成 DreamControl 与 CLoSD 原论文复现。已完成 DreamControl 的 Kick、Punch 两个控制任务，其余九项未启动；CLoSD 已完成缩容评估，尚未完整复现。当前服务器没有正式训练或评估在运行。完整验收边界与下一步顺序见[基线复现进度](reports/03_baseline_summary.md)。

同日新增两项冻结策略的 **Isaac Lab → MuJoCo 首轮 sim-to-sim**：Kick 到达虚拟目标 54/66（81.8%），到达且完整 10 秒不摔倒 48/66（72.7%）；Punch 对应 42/100（42.0%）、33/100（33.0%）。无重新训练或额外 IK，两个策略的观测/动作与原检查点已核对。每条参考单独测试，口径不同于官方自动重置评估，尚未通过扰动鲁棒性或实机验收。见[报告](reports/10_Kick_Punch_MuJoCo_sim2sim_20261001.md)与[真实 MuJoCo 视频](media/sim2sim_mujoco/results.html)。

随后完成 **2,324 次冻结策略扰动／实体接触测试**：覆盖实际接触摩擦 0.5/1.0/1.5、质量 ±10%、力矩上限 −10%、左右 20/40 N 推力及固定打击靶。原位置靶的合格打击率 Kick 39/66（59.1%）、Punch 42/100（42.0%）；打击且完整 10 秒站稳分别仅 9/66（13.6%）、33/100（33.0%）。高摩擦条件下两项明显退化，不能宣称鲁棒性通过。首轮地面摩擦合并配置问题已留档并修正后完整重跑；最终 V2 有逐接触摩擦核验。[完整报告](reports/11_Kick_Punch_robustness_contact_20261001.md)与[12 段场景视频](media/mujoco_robust_contact/results.html)。这不增加论文训练任务完成数，也不代表实机打击或创新点验证。

## 当前可核查结果

| 实验 | 数据与预算 | 评估口径 | 当前结果 |
| --- | --- | --- | --- |
| DreamControl 官方 Kick 基线 | 官方 66 条轨迹，8192 环境，2000 轮，seed 42 | 官方 `play_eval.py`，1000 环境、500 帧 | **85.3%** |
| 旧 66 条策略在新动作留出集 | 1024 环境训练；32 条 seed42 留出轨迹 | 1000 环境、500 帧 | **0%** |
| 98 条增强组在同一留出集 | 原 66＋新增 32 条，1024 环境，seed 123 | 同一 32 条留出、1000 环境、500 帧 | **0%** |
| CLoSD 多任务闭环微调 | 官方预训练控制器，512 环境，增加 4000 epoch 后保存检查点 | 同口径微调前／后评估 | REACH 100%→0%、共用 STRIKE 87%→5%、SIT 77%→7%、GET_UP 56%→69%；训练在保存后异常退出 |
| I1 脚部约束后处理诊断 | 同一组 100 个 Kick 约束，生成种子 42/43/44；仅修改成功重定向轨迹的右腿 | 外部目标几何误差、固定策略仿真、接触与耗时 | 几何误差下降，但修正后跟踪显著退化；**完整 I1 Gate 未通过** |

**85.3% 与两个 0% 不能直接比较数据增益**：训练环境数、种子和测试动作不同。增强组的同口径留出集实验尚未显示成功率收益。CLoSD 512 环境缩容微调并非论文 4096 环境的严格同配置复现；Kick 与 I1 的单动作类别结果也不能代表通用生成。准确的时间线、失败尝试与限制见 [DreamControl 记录](reports/02_dreamcontrol_reproduction.md)、[CLoSD 记录](reports/01_closd_reproduction.md)及 [I1 执行与质量诊断](reports/05_I1_dynamics_contact_semantics_efficiency.md)。

## 内容

- `plans/R3_experiment_plan.md`：当前通用条件动作生成研究方案与阶段门。
- `configs/`：项目本地的版本化环境规格；上游 Isaac Sim、Isaac Gym、DreamControl 和 CLoSD 均由各自官方来源获取。
- `scripts/`：复现、资产核验、条件动作生成、质量审计、训练／评估与监控脚本。
- `results/`：机器可读的实验口径与结果索引。
- `reports/04_I1_feasibility.md`、`reports/05_I1_dynamics_contact_semantics_efficiency.md`：空间条件、机器人运动学、固定策略仿真、接触代理、语义代理及生成耗时的阶段证据；均未宣称完整 I1 成立。
- `reports/03_dreamcontrol_multitask_baselines.md`：其余十任务的参考数据、严格复现口径与当前阶段状态。
- `media/`：已完成评估的小型视频。
- `checkpoints/`：DreamControl 实验的控制策略检查点，使用 Git LFS；**不含**受单独许可约束的 CLoSD、SMPL、AMASS 或 HumanML3D 资产。

DreamControl 官方代码来源记录为提交 `ec28b508313d6837de52c68247fbc8cf81fe4935`，但当前服务器的 `third_party/DreamControl` 目录不是独立 Git checkout，不能仅凭该目录再次证明完整代码版本；本轮关键源文件哈希见 `results/r3_i1_provenance_manifest_v2.json`。CLoSD 官方代码来源记录为提交 `d8950bdb78312fcd2844f336b3fc079e7368c87f`。新任务的服务器工作目录为 `/data/code/ouyanghao/oyh/conditional-human-motion-generation`，DreamControl 源码、参考动作和后续日志独立保存；通用 Isaac Sim 安装与 Python 依赖仍作为只读共享工具使用。此前运行的 CLoSD 训练保持原路径 `/data/code/ouyanghao/oyh/humanoid-combat-motion` 直到完成，避免移动文件中断进程；其结果再归档到本仓库。

官方 DreamControl 的论文报告 11 种任务。其余十项任务的公开参考轨迹已逐文件校验，哈希见 `results/official_other_tasks_assets_manifest.json`；这只表示输入数据就绪，**不是训练完成**。首项 Punch 在第 1618 轮因图形串流故障停止，后来从第 1600 轮检查点续训至第 1999/2000 轮并完成正式评估，成功率为 87.2%。旧顺序队列未接续其余九项；需用训练与画面监控分离的入口恢复剩余任务。不能用 Kick 或 Punch 的结果代表其余任务。[论文与任务表](https://genrobo.github.io/DreamControl/static/paper/DreamControl_arxiv.pdf)。

## 数据与许可

本仓库不含官方大型动作库、模型权重或机器人资产。CLoSD 使用的 SMPL、AMASS 与 HumanML3D 需遵守各自许可；因此其原始数据及由其衍生的控制器权重不会未经单独核实直接发布。DreamControl 的官方资产和生成轨迹请按原项目说明获取并校验哈希。完整训练日志与较大的中间产物保留在服务器，`results/` 记录必要的路径、参数和摘要。

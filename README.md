# Conditional Human Motion Generation

本仓库记录**条件驱动人体运动生成**课题的可复现实验。研究流程是条件输入 → 生成人体运动 → 物理／接触修正 → 机器人参考动作 → 固定下游执行与反馈。机器人跟踪用于检验生成动作是否可执行，并非本课题的唯一研究对象。本仓库与原人形机器人格斗项目分开。

## 当前可核查结果

| 实验 | 数据与预算 | 评估口径 | 当前结果 |
| --- | --- | --- | --- |
| DreamControl 官方 Kick 基线 | 官方 66 条轨迹，8192 环境，2000 轮，seed 42 | 官方 `play_eval.py`，1000 环境、500 帧 | **85.3%** |
| 旧 66 条策略在新动作留出集 | 1024 环境训练；32 条 seed42 留出轨迹 | 1000 环境、500 帧 | **0%** |
| 98 条增强组在同一留出集 | 原 66＋新增 32 条，1024 环境，seed 123 | 同一 32 条留出、1000 环境、500 帧 | **0%** |
| CLoSD 多任务闭环微调 | 官方预训练控制器，512 环境，计划增加 4000 epoch | 尚在训练，最终多任务评估未出 | **进行中** |

**85.3% 与两个 0% 不能直接比较数据增益**：训练环境数、种子和测试动作不同。增强组的同口径留出集实验尚未显示成功率收益。完整 DreamControl 上游生成—重定向—修正流程和 CLoSD 最终 Gate 仍需分别核验。准确的时间线、失败尝试与限制见 [DreamControl 记录](reports/02_dreamcontrol_reproduction.md)和 [CLoSD 记录](reports/01_closd_reproduction.md)。

## 内容

- `plans/R3_experiment_plan.md`：当前通用条件动作生成研究方案与阶段门。
- `configs/`：项目本地的版本化环境规格；上游 Isaac Sim、Isaac Gym、DreamControl 和 CLoSD 均由各自官方来源获取。
- `scripts/`：复现、资产核验、条件动作生成、质量审计、训练／评估与监控脚本。
- `results/`：机器可读的实验口径与结果索引。
- `reports/03_dreamcontrol_multitask_baselines.md`：其余十任务的参考数据、严格复现口径与当前阶段状态。
- `media/`：已完成评估的小型视频。
- `checkpoints/`：DreamControl 实验的控制策略检查点，使用 Git LFS；**不含**受单独许可约束的 CLoSD、SMPL、AMASS 或 HumanML3D 资产。

DreamControl 官方代码固定提交 `ec28b508313d6837de52c68247fbc8cf81fe4935`；CLoSD 官方代码固定提交 `d8950bdb78312fcd2844f336b3fc079e7368c87f`。新任务的服务器工作目录为 `/data/code/ouyanghao/oyh/conditional-human-motion-generation`，DreamControl 源码、参考动作和后续日志独立保存；通用 Isaac Sim 安装与 Python 依赖仍作为只读共享工具使用。此前运行的 CLoSD 训练保持原路径 `/data/code/ouyanghao/oyh/humanoid-combat-motion` 直到完成，避免移动文件中断进程；其结果再归档到本仓库。

官方 DreamControl 的论文报告 11 种任务。其余十项任务的公开参考轨迹已逐文件校验，哈希见 `results/official_other_tasks_assets_manifest.json`；这只表示输入数据就绪，**不是训练完成**。`scripts/r3_dreamcontrol_official_other_tasks_queue.sh` 为顺序训练／评估队列，必须先通过本地查看真实训练仿真的图形 Gate 才能启动。每项任务独立核验正式训练预算、成功率和视频，不能用 Kick 的结果代表其余任务。[论文与任务表](https://genrobo.github.io/DreamControl/static/paper/DreamControl_arxiv.pdf)。

## 数据与许可

本仓库不含官方大型动作库、模型权重或机器人资产。CLoSD 使用的 SMPL、AMASS 与 HumanML3D 需遵守各自许可；因此其原始数据及由其衍生的控制器权重不会未经单独核实直接发布。DreamControl 的官方资产和生成轨迹请按原项目说明获取并校验哈希。完整训练日志与较大的中间产物保留在服务器，`results/` 记录必要的路径、参数和摘要。

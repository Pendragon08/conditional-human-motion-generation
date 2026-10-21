# PhyGile 部分复现：GMT、262D 与真实仿真接口

**状态：部分复现。** 此目录保存固定版本官方代码、仿真适配、实际运行日志及小规模结果。完整文本扩散生成和 physics-prefix 闭环组件未在检查的官方仓库中找到；本目录不能替代完整 PhyGile，也不能作为论文主表数值已复现的证明。

- [最新课程评估、262D 与真实闭环接口报告](reports/PhyGile课程评估与262D闭环补全报告.md)
- [上一轮课程修复与物理前缀报告](reports/PhyGile课程修复与物理前缀推进报告.md)
- [2000 轮 PPO 训练与自实现生成核心报告](reports/PhyGile续训与生成核心报告.md)
- [上一轮 500 轮 PPO 训练报告](reports/PhyGile复现实验报告.md)
- [范围、缺失材料与论文用途](docs/reproduction_scope.md)
- [环境及独立运行验证](docs/environment.md)
- [来源和逐文件哈希](upstream_manifest.json)
- [原始官方代码](https://github.com/Baojch/phygile_tracking/tree/d48621620f522f5c901ef378d57db8df92d2888d)

## 本轮新增交付

新增独立课程评估服务和训练调用、版本化 262D 编解码，以及实际扩散采样—G1 跟踪—拒绝/重放接口。完整运行约定见 [completion_invocation](docs/completion_invocation.md)，站立边界与确定性修复见 [terminal_recovery_invocation](docs/terminal_recovery_invocation.md)。先前失败的末端补测原样保留。不可靠的旧评估入口保持禁用；完整课程训练未开展。生成器为随机权重工程诊断，闭环 PPO 未执行，不能当作语义生成能力。

公开跟踪器从固定 500 轮 PPO 检查点续训至累计 2000 轮，并完成 8 组固定最终检查点评估。测试完整片段成功为 **26/128（20.31%）**，平均持续 1.501 秒；测试仅一个来源组、一个训练种子，不是论文主表复现。复跑见 [续训协议](docs/continuation_2000.md)，大文件见 [2000 轮 PPO 训练附件](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/phygile-tracking-2000updates-20261017)。

经用户选择，缺失生成部分按论文独立编写，保存于 [reimplementation](reimplementation/README.md)，与 `upstream` 官方代码分开。先前的数值、真实 DistilBERT 与人工张量训练入口检查保留；本轮增加真实机器人编解码和仿真接口。完整动作生成器没有用真实配对数据训练。示例训练参数与未公开的数值约定均为本实现选择，不能当作作者设置。

2026-10-18 已取得 HumanML3D 官方公开文本、划分和动作来源索引，见 [配对数据准备状态](docs/paired_data_status_20261018.md)。对应原始 AMASS 动作、所选重定向路径的人体模型及 G1 配对动作仍缺；没有以无配对文本的武术片段代替完整生成训练。已有独立编解码，作者一致性尚待确认；ASFO 镜像、完整课程训练、真实生成分布上的 PPO 适配和正式生成评估仍待补齐。原始记录、代码/输入哈希与独立审查均保留。

2026-10-21 已生成完整逐 ID 时间对齐计划、去重来源清单并通过 8 项时间边界检查，见 [AMASS 获取与对齐准备](docs/amass_acquisition_and_alignment_20261021.md)。SSM 已取得并核验 30 条原始 SMPL-X 动作，见 [SSM 核验记录](docs/SSM_download_status_20261021.md)；人体模型及其余子集仍缺，本轮未新增 G1 配对动作或启动生成器训练。

计数更正与参考来源见 [更正说明](docs/accounting_terms_erratum.md)：旧 updates 字段表示 PPO 轮次，每轮 20 次 Adam 步进，累计 40,000 次；外部运动参考不是实测机器人真值。

## 从头运行 500 轮 PPO 训练试验

Linux、Isaac Sim 4.5.0、Isaac Lab 2.1.0、Python 3.10、PyTorch 2.5.1+cu118、RSL-RL 2.3.1。精确依赖见 `configs/env-spec.json`。`scripts/run_isaac.sh` 支持 `ISAAC_SIM_ROOT` 和 `ISAAC_LAB_ROOT` 环境变量；已有服务器通过默认路径复用安装，未修改共享环境。

机器人描述须按 `upstream/README.md` 从官方说明链接获取，放在 `upstream/source/whole_body_tracking/whole_body_tracking/assets/unitree_description`。已有服务器使用只读符号链接。

动作应置于 `data/train` 和 `data/test`。当前实验是已有 KungfuAthleteBot 来源 G1 参考的 20/4 片段子集，每个文件带有明确关节和刚体名称。`results/data_inventory.json` 和 `results/data_provenance.json` 给出输入哈希与来源链。它不是作者论文数据集。实验附件 `phygile_pilot_inputs.tar.gz` 包含这 24 个处理后的输入及许可说明。

仓库已带归档结果，复跑时先创建不包含旧结果的新目录，避免覆盖记录：

```sh
python3 scripts/create_run_workspace.py --destination /path/to/new-phygile-run
cd /path/to/new-phygile-run
python3 scripts/unpack_inputs.py /path/to/phygile_pilot_inputs.tar.gz
```

按上文配置仿真安装与机器人资产后，在这个新目录中运行：

```sh
bash scripts/run_isaac.sh scripts/kernel_witness.py
bash scripts/run_isaac.sh scripts/data_inventory.py
bash scripts/smoke.sh
python3 scripts/run_recorded.py --name pilot_train20_nostate_seed42 --timeout 7200 -- bash scripts/train_pilot.sh
python3 scripts/evaluate_suite.py
bash scripts/run_isaac.sh scripts/summarize_results.py
```

`run_recorded.py` 拒绝覆盖同名结果目录。重新运行时应在新的工作副本中运行，或使用新的记录名称并同步评估脚本的固定收据路径。从头训练脚本为 1024 个环境、500 轮 PPO、seed 42，使用官方无状态估计观测任务。

**当前配置只激活第一个专家。** 官方 Phase A 默认从 1 个专家开始，随课程增加；本实验没有作者难度标签，关闭课程。这是起始跟踪训练验证，不能用于评估完整 MoE/动态专家扩展的贡献。

另一个已核查问题：公开加载器忽略 YAML 的 `rewards` 段。实际 global anchor position 奖励为源码默认权重 0.5/std 0.25，未应用 YAML 中的 0.8/0.2。有效参数应查看运行保存的 `env.yaml`；报告已记录此差异，未在实验中途修改奖励。

## 固定评估协议

上一轮评估 1 轮和 500 轮 PPO 检查点，本轮对固定 2000 轮检查点沿用同一协议：训练片段 0/8/10/14 和所有 4 个测试片段，每片段 32 次并行首回合。测试来源分组与训练不重叠。保持训练失败阈值，从第 0 帧开始，保留初始扰动和物理随机化，关闭间隔推力及观测噪声。

指标在物理步后、自动重置前采集，与当前命令帧比较。完整片段成功须跑完全部帧且未触发失败；误差只累计至终止，必须结合成功率和时长解读。`raw_pre_reset.npz` 保留原始状态用于重算；`per_step_metrics.csv` 和汇总在结果目录中。该协议不是论文完整基准协议的替代。

本报告的 MPJPE 使用世界坐标，与官方根部 XY/偏航对齐误差不同；训练日志的 `Train/success` 也不等于此处的完整片段成功率。完整原始状态在实验附件中，Git 仅保存较小的日志、逐步指标和汇总。

## 完整实验准备

`prepare_full_config.py` 可把公开 Phase A/B 配置中的机器路径替换为提供的数据与标签路径。该脚本只做路径准备，不证明数据与论文等价，也不自动启动大规模训练。Phase B 需要合适的 Phase A 检查点；完整 PhyGile 还缺真实配对数据、已训练生成器和生成分布上的闭环 PPO 适配。详见范围说明。

## 许可

官方源码许可与文件头完整保留于 `upstream/LICENCE` 和源文件。本次新编写的适配、记录与评估脚本使用 MIT 许可（同 `upstream/LICENCE` 条款）。机器人描述和外部数据各自遵循其原许可。原始数据不混入代码版本历史。

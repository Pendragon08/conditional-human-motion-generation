# PhyGile 公开 GMT 跟踪代码复现

**状态：部分复现。** 此目录保存固定版本官方代码、仿真适配、实际运行日志及小规模结果。完整文本扩散生成和 physics-prefix 闭环组件未在检查的官方仓库中找到；本目录不能替代完整 PhyGile，也不能作为论文主表数值已复现的证明。

- [复现实验报告](reports/PhyGile复现实验报告.md)
- [范围、缺失材料与论文用途](docs/reproduction_scope.md)
- [环境及独立运行验证](docs/environment.md)
- [来源和逐文件哈希](upstream_manifest.json)
- [原始官方代码](https://github.com/Baojch/phygile_tracking/tree/d48621620f522f5c901ef378d57db8df92d2888d)

## 运行

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

`run_recorded.py` 拒绝覆盖同名结果目录。重新运行时应在新的工作副本中运行，或使用新的记录名称并同步评估脚本的固定收据路径。训练脚本为 1024 个环境、500 次更新、seed 42，使用官方无状态估计观测任务。

**当前配置只激活第一个专家。** 官方 Phase A 默认从 1 个专家开始，随课程增加；本实验没有作者难度标签，关闭课程。这是起始跟踪训练验证，不能用于评估完整 MoE/动态专家扩展的贡献。

另一个已核查问题：公开加载器忽略 YAML 的 `rewards` 段。实际 global anchor position 奖励为源码默认权重 0.5/std 0.25，未应用 YAML 中的 0.8/0.2。有效参数应查看运行保存的 `env.yaml`；报告已记录此差异，未在实验中途修改奖励。

## 评估

评估 1 次更新和 500 次更新的检查点；分别选定训练片段 0/8/10/14 和所有 4 个测试片段，每片段 32 次并行首回合。测试来源分组与训练不重叠。保持训练失败阈值，从第 0 帧开始，保留初始扰动和物理随机化，关闭间隔推力及观测噪声。

指标在物理步后、自动重置前采集，与当前命令帧比较。完整片段成功须跑完全部帧且未触发失败；误差只累计至终止，必须结合成功率和时长解读。`raw_pre_reset.npz` 保留原始状态用于重算；`per_step_metrics.csv` 和汇总在结果目录中。该协议不是论文完整基准协议的替代。

本报告的 MPJPE 使用世界坐标，与官方根部 XY/偏航对齐误差不同；训练日志的 `Train/success` 也不等于此处的完整片段成功率。完整原始状态在实验附件中，Git 仅保存较小的日志、逐步指标和汇总。

## 完整实验准备

`prepare_full_config.py` 可把公开 Phase A/B 配置中的机器路径替换为提供的数据与标签路径。该脚本只做路径准备，不证明数据与论文等价，也不自动启动大规模训练。Phase B 需要合适的 Phase A 检查点；完整 PhyGile 还缺生成器、权重及前缀闭环实现。详见范围说明。

## 许可

官方源码许可与文件头完整保留于 `upstream/LICENCE` 和源文件。本次新编写的适配、记录与评估脚本使用 MIT 许可（同 `upstream/LICENCE` 条款）。机器人描述和外部数据各自遵循其原许可。原始数据不混入代码版本历史。

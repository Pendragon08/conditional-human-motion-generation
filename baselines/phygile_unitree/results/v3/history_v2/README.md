# PhyGile 的 Unitree 框架适配与跨仿真记录

已实现 **G1 跟踪器** 的 Unitree RL Lab 自定义任务和 Unitree RL Mjlab 冻结策略入口。保留论文专用 MoE/MoEPPO、616D 观测、29D 动作及经验归一化，不替换成框架默认 MLP/PPO。此分支是自行实现的框架适配，官方仓库是依赖，不代表论文作者提供了这份适配。

2026-10-22 V2：用户已授权无界面续训；2 次 PPO / 40 次 Adam 的恢复训练验证通过，另一个 500 次迭代任务正在运行。两个 Unitree 入口均完成实际扩散采样、262D 解码、仿真执行、反馈、拒绝及回放恢复的工程验证；未训练生成器的候选均被拒绝，不能报告稳定语义生成。足部摩擦对齐后仍仅完成 1/4 个 MuJoCo 诊断片段，稳定性尚未达标。

- [V2 无界面续训与双仿真闭环报告](reports/PhyGile_无界面续训与双仿真闭环报告_20261022.md)
- [V2 执行记录与独立见证](results/v2/)

- [实验报告](reports/PhyGile_Unitree适配与跨仿真报告_20261022.md)
- [环境与独立执行见证指令](docs/ENVIRONMENT.md)
- [控制及模型契约](results/v1/contract.json)
- [逐片段结果及输入数值核验](results/v1/crosssim_summary.json)
- [实验状态](EXPERIMENT_TRACKER.md)

## 固定官方代码

先从本仓库根目录执行以下命令，两个目标目录必须不存在。脚本保留官方源码，不修改现有框架目录。

```text
python baselines/phygile_unitree/scripts/prepare_unitree_vendor.py --framework lab --destination baselines/phygile_unitree/vendor/unitree_rl_lab
python baselines/phygile_unitree/scripts/prepare_unitree_vendor.py --framework mjlab --destination baselines/phygile_unitree/vendor/unitree_rl_mjlab
```

Isaac 运行入口是 `scripts/run_isaac.sh`，默认使用原服务器 PhyGile 工程的只读源码、配置和检查点。参数示例（在 `baselines/phygile_unitree` 目录中执行；替换检查点与全新输出目录）：

```text
bash scripts/run_isaac.sh scripts/lab_run.py --mode evaluate --checkpoint /path/to/model_1999.pt --motion-file /path/to/test_motion.npz --output /path/to/new_result --num-envs 1 --device cuda:1 --enable_cameras --kit_args="--/renderer/multiGpu/enabled=false --/renderer/activeGpu=0"
```

上述示例为 V1 图形模式；该模式下 `DISPLAY=:10` 必须有效，仅 GPU0 可呈现。V2 无界面模式按用户明确授权执行，入口要求 `--headless --execution-consent /path/to/headless_authorization_20261022.json`；准确执行与验收见 `docs/HEADLESS_V2_INVOCATION.md`。`PHYGILE_PARENT` 可覆盖原工程位置，依赖路径见运行脚本。

Mjlab 的本机已部署指令、确切包版本和 Windows 英文路径入口见环境文档。移到其他机器时需按规格新建独立环境，准备有权使用的 `private_assets`；不得将 AMASS/人体模型原始资产放入公开仓库。`analyze_crosssim.py` 可从两套引擎的私有原始状态重新计算全部指标；公开保存指标、来源哈希、执行回执、合成夹具及日志。

## 范围

V1 历史结果为 Isaac 完成2/4、MuJoCo完成1/4；当时没有新训练。V2 已有实际训练更新和双引擎闭环工程迁移，完整500次续训尚未完成。四段149帧、50Hz动作已用于开发诊断，每配置仅一个种子；不能报告论文全量性能、泛化或真机结果。原数据、原权重和 ReactiveBFM 官方规划器/跟踪器工程均保留。

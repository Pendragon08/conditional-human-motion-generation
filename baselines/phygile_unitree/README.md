# PhyGile 的 Unitree 框架适配与跨仿真记录

已实现 **G1 跟踪器** 的 Unitree RL Lab 自定义任务和 Unitree RL Mjlab 冻结策略入口。保留论文专用 MoE/MoEPPO、616D 观测、29D 动作及经验归一化，不替换成框架默认 MLP/PPO。此分支是自行实现的框架适配，官方仓库是依赖，不代表论文作者提供了这份适配。

四个短测试片段中，Isaac 路线完整完成 2/4，MuJoCo 路线完成 1/4；这是一次存在失稳的跨仿真小规模验证。新 PPO 续训尚未启动：用户暂时无法查看服务器窗口，项目要求先验证本地连接。

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

`DISPLAY=:10` 必须指向有效的服务器图形会话；本次仅 GPU0 可呈现该 Xorg 窗口。服务器的 `PHYGILE_PARENT` 可覆盖原工程位置，依赖路径见运行脚本。冻结推理可用独立进程验证；新训练还需真实本地查看回执。

Mjlab 的本机已部署指令、确切包版本和 Windows 英文路径入口见环境文档。移到其他机器时需按规格新建独立环境，准备有权使用的 `private_assets`；不得将 AMASS/人体模型原始资产放入公开仓库。`analyze_crosssim.py` 可从两套引擎的私有原始状态重新计算全部指标；公开保存指标、来源哈希、执行回执、合成夹具及日志。

## 范围

本次没有新增训练权重，没有在新框架重新训练完整 MoE，也未将既有 physics-prefix 生成—执行闭环迁入这两个入口。四个测试片段均为 149 帧、50Hz、每段每引擎一个种子；不能据此报告论文全量性能、泛化或真机结果。原数据、原权重和 ReactiveBFM 官方规划器/跟踪器工程均保留。

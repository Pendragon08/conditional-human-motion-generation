# PhyGile 的 Unitree 框架适配与跨仿真记录

已实现 **G1 跟踪器** 的 Unitree RL Lab 自定义任务和 Unitree RL Mjlab 冻结策略入口。保留论文专用 MoE/MoEPPO、616D 观测、29D 动作及经验归一化，不替换成框架默认 MLP/PPO。此分支是自行实现的框架适配，官方仓库是依赖，不代表论文作者提供了这份适配。

2026-10-22 V3：已按用户授权完成500次无界面PPO续训、10000次Adam更新，双引擎闭环在父策略及新策略上均完成工程执行。新策略Isaac完整完成数从2/4降为1/4，MuJoCo仍为1/4；摩擦与源碰撞形状对照未解决退化。继续保留父策略为当前基线。未训练生成器的候选全部被拒绝，不能报告稳定语义生成或完整论文复现。

- [本轮最终报告](reports/PhyGile_Unitree续训500次与闭环迁移最终报告_20261022.md)
- [完成结果、权重来源与逐片段对照](results/v3/summary.json)
- [V3全部证据索引](results/v3/evidence_index.json)

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

V1没有新训练，V2是进行中快照，V3完成了本轮500次续训与后续评估。四段149帧、50Hz动作已用于开发诊断，每配置仅一个种子；不能报告论文全量性能、泛化或真机结果。生成闭环已接通但生成提交帧数为0。原数据、原权重和ReactiveBFM官方规划器/跟踪器工程均保留。

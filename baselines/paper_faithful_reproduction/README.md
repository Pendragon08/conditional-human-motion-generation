# PhyGile / ReactiveBFM：论文对齐复现

**状态：进行中，尚未完成两篇论文的完整性能复现。**

- [完整中文进展报告](reports/reproduction_progress_20261023.md)：原文与代码差异、训练进度、失败结果和未完成部分。
- [视频、报告离线包与原始实验记录](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/paper-reproduction-progress-20261023)。下载 `report_and_videos.zip` 后解压，打开 `复现报告与视频.html`。
- [运行环境和独立复跑](ENVIRONMENT.md)。环境说明中的机器绝对路径是本次实验位置，其他机器需自行映射。

## 本轮实际完成

1. 固定原文、官方仓库版本和源文件哈希，逐项审核算法及训练参数。
2. PhyGile 专家网络修正为论文的 `[2048,1024,512]`，补齐课程与 PPO 参数；使用官方 MoE 类完成前向/反向检查。这不是 RL 训练结果。配置保留未获得数据与难度 CSV 的明确占位。
3. ReactiveBFM 保留运行中的官方训练与 ScaleBFM 跟踪器，恢复后的训练继续执行。独立兼容层修复异步分支和 Torch 2.5 编译问题，完成实际仿真、轨迹记录及编译一致性检查。
4. 保存全部失败尝试。早期 ReactiveBFM 检查点完成 500 次控制、20 次有效重规划，但 37 帧超出近似跟踪失败阈值；不能报告为成功率达标。
5. 由实际保存的状态制作两段视频。PhyGile 四动作在两种仿真中均只有一个完整执行，不选取成功动作冒充整体效果。

## 代码入口

| 文件 | 用途 |
|---|---|
| `scripts/build_paper_configs.py` | 从固定官方配置生成论文对齐配置和差异清单 |
| `scripts/verify_phygile_architecture.py` | 在官方 MoE 类上检查专家结构、前向及反向 |
| `scripts/run_reactive_async.py` | 固定官方实现之上的异步执行兼容层；记录真实状态和源文件哈希 |
| `scripts/analyze_reactive_recording.py` | 对齐真实状态与当前计划，计算全回合误差和失败帧 |
| `scripts/render_phygile_records.py` | 显示全部四个既有实际轨迹；记录根姿态重建误差 |
| `scripts/document_witness.py` | 文档独立复跑，检查执行、有限值、源文件及训练状态 |
| `scripts/relocate_preserved_archive.py` | 本次单次文件迁移记录：完整复制核验后保留原路径链接；不可盲目重复运行 |

配置生成示例（需要 PyYAML，两个官方仓库固定在 `sources/upstream_pins.json` 的版本）：

```text
python scripts/build_paper_configs.py --phygile-upstream /path/to/phygile_tracking --reactive-upstream /path/to/ReactiveBFM --output-root /path/to/new-output
```

其他入口先查看 `--help`。执行环境、检查点与部署源代码路径必须与实际资产匹配。`results/reactive_async_v4/` 和独立复跑记录包含真实调用与校验信息。

## 完成条件与后续顺序

先完成公开可获得数据与论文算法对应的主线，再将 Unitree 迁移作为补充。PhyGile 仍需全量可获得配对数据、真实物理过滤、课程难度标签、论文网络的两个训练阶段、生成器训练及生成分布上的闭环 PPO。不能将 66 条 SSM 序列或未训练模型替代这些阶段。原文中的私有 3 小时语料不可从公开代码恢复，报告必须保留这一不可消除的差异。

ReactiveBFM 当前公开代码训练分支继续至 400k/1M 里程碑；不得中断现有训练。后续增加明确区分的 20/40 窗口对照、指令切换分支、独立评价和实时推理验证。原文表 4 与发布程序存在冲突，按报告保留并核查，不能悄悄选择参数后声称完全一致。

每个新结果须关联检查点、配置、源版本、完整执行日志与失败记录。只在取得新测量后更新指标和视频。大检查点由已存在的备份程序保存；本轮不删除数据集或实验记录。受许可限制的 AMASS/SMPL 原始资产不进入公开 Release。

## 数据与视频可追溯性

`results/public_artifact_manifest.json` 列出发布文件的大小和 SHA256；GitHub Release 的资产摘要须与之匹配。`async_evidence_v1_v4.tar.gz` 保存四次尝试的执行日志、兼容层快照与实际生成/仿真状态，不包含 AMASS 原始训练资产。视频按仿真时间播放，不代表墙钟实时性。

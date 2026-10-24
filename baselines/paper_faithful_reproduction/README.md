# PhyGile / ReactiveBFM：论文对齐复现

**状态：进行中，尚未完成两篇论文的完整性能复现。**

- [最新状态：10-24 11:49连接异常与备份复核](reports/heartbeat_20261024_1149.md)：SSH四次均未成功，当前训练/磁盘/新检查点状态未知，不能断言训练停止。已有13,001检查点等6件资产的GitHub摘要与本地完整SHA256复核通过；本轮无新训练或视频，等待连接恢复。
- [22:46检查及后续：13,001检查点与实际闭环](reports/heartbeat_20261023_2246.md)：模型/EMA/优化器三端核验；500控制、20有效重规划，0/500帧违规，仅本次近似诊断通过。原训练继续，下一次安全保存需新增存储；MANO入口恢复，等待登录及兼容模型。
- [19:45检查及后续：12,001检查点闭环与存储维护](reports/heartbeat_20261023_1945.md)：新模型/优化器本地与GitHub备份通过；500次控制、20次有效重规划，1/500帧违规，仍判失败。两个旧源码包在恢复核验后清理，受保护数据/检查点/记录不删，训练继续；需新增持久存储。
- [最新：配对动作辅助物理筛选与完整训练缺口](reports/paired_filter_progress_20261023.md)：40条、17个录制；冻结外部ScaleBFM整段执行32/40，21/40通过辅助条件；三个seed轨迹一致，非120个独立试验。全部失败和实际回放保留，未替代论文GMT过滤，完整生成器未启动。
- [12:27后续：修复PhyGile生成/控制时钟](reports/heartbeat_20261023_1227.md)：训练检查点绑定真实特征FPS，20→50Hz固定时间转换与规范坐标还原；6项反例、40条真实配对及独立文档复跑通过。服务器小网络两步CPU测试通过，不计作真实生成器训练；ReactiveBFM续训11,197次，未中断。
- [最新：磁盘清理与复现继续推进](reports/storage_cleanup_and_progress_20261023.md)：106个闲置缓存/旧传输包在备份核验后清理，释放3.26GiB；ReactiveBFM新检查点11,001已完整备份，真实闭环5/500帧超近似阈值，仍判失败。PhyGile补录3案例终止项，全部状态与旧记录逐位一致，未放宽判据。
- [09:28：40条配对动作实际跟踪诊断与检查点备份](reports/heartbeat_20261023_0928.md)：PhyGile既有小样本策略完整执行2/40，失败记录全部保留，不生成论文课程标签；ReactiveBFM10,061次检查点的模型及优化器已完整备份至本地和GitHub，续训采样为10,660次有效更新。
- [检查点10,061：真实闭环评价与视频](reports/checkpoint_10061_progress.md)：候选续训已保存新模型；500次控制、20次有效重规划、33/500帧超过近似失败阈值，仍判失败。新视频和完整轨迹记录已保存。
- [2026-10-23 06:23 检查与数值恢复](reports/heartbeat_20261023_0623.md)：v3 在16,871次更新后自行异常退出，最近可恢复检查点为10,041次；完整故障记录及精度对照已保存，恢复状态见该报告。
- [同轮备份重试补充](reports/backup_followup_20261023.md)：首次复制遇到连接超时，错误已保留；后续只复制任务使用 `_retry1` 独立目录。
- [2026-10-23 03:26 后续进展](reports/heartbeat_20261023_0326.md)：训练检查及 PhyGile 时基一致子集，保留40条、暂不纳入26条。
- [完整中文进展报告](reports/reproduction_progress_20261023.md)：原文与代码差异、训练进度、失败结果和未完成部分。
- [视频、报告离线包与原始实验记录](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/paper-reproduction-progress-20261023)。下载 `report_and_videos.zip` 后解压，打开 `复现报告与视频.html`。
- [运行环境和独立复跑](ENVIRONMENT.md)。环境说明中的机器绝对路径是本次实验位置，其他机器需自行映射。

## 本轮实际完成

1. 固定原文、官方仓库版本和源文件哈希，逐项审核算法及训练参数。
2. PhyGile 专家网络修正为论文的 `[2048,1024,512]`，补齐课程与 PPO 参数；使用官方 MoE 类完成前向/反向检查。这不是 RL 训练结果。配置保留未获得数据与难度 CSV 的明确占位。
3. ReactiveBFM 保留官方训练流程与 ScaleBFM 跟踪器。v3 训练随后自行出现数值异常，现按最新报告验证恢复候选；独立异步兼容层已完成实际仿真、轨迹记录及编译一致性检查。
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
| `scripts/gate_phygile_sample_clock.py` | 独立生成时基一致清单，排除标头冲突和抽帧引入的播放速度变化；原数据不变 |
| `scripts/screen_phygile_pairs_mjlab.py` | 将时基一致的262D配对特征解码并执行实际MuJoCo首回合诊断，保留全部失败；不产生作者课程标签 |
| `scripts/render_paired_screen_records.py` | 逐帧回放全部配对动作诊断的实际观察，受限衍生视频仅保存在本地 |
| `scripts/probe_cross_corpus_lengths.py` | 只读核查跨语料同名动作长度；不把ID匹配当作时间段对齐 |
| `scripts/probe_phygile_termination.py` | 只读记录实际pre-reset终止项与手脚高度误差；3个案例全部原始数组与旧记录逐位一致 |
| `scripts/verify_phygile_generator_clock.py` | 数值时钟、规范坐标与前缀保留验证；真实40条仅用于数值检查，不执行仿真或训练 |
| `scripts/cleanup_backed_unused.py` | 仅清理精确白名单中的闲置缓存/旧传输包；要求先验证本地与GitHub备份，并检查所有者、文件身份及活动占用 |
| `scripts/diagnose_reactive_gradients.py` | 固定检查点和真实批量，对照注意力精度与激活重算；不执行优化器更新 |
| `recovery_v4/` | FP64 注意力、FP32 参数的独立恢复候选；固定源码与精确文档见证，不能视为已验证长时间稳定 |
| `scripts/copy_recovery_checkpoint.py` | 新检查点只复制、核验并上传 GitHub；不删除服务器文件 |
| `scripts/relocate_preserved_archive.py` | 本次单次文件迁移记录：完整复制核验后保留原路径链接；不可盲目重复运行 |

配置生成示例（需要 PyYAML，两个官方仓库固定在 `sources/upstream_pins.json` 的版本）：

```text
python scripts/build_paper_configs.py --phygile-upstream /path/to/phygile_tracking --reactive-upstream /path/to/ReactiveBFM --output-root /path/to/new-output
```

使用已有 SSM 配对数据前，应先运行时基检查；输出目录必须尚不存在：

```text
python scripts/gate_phygile_sample_clock.py --dataset /path/to/ssm_batch_v1 --phygile-code /path/to/baselines/phygile --output /path/to/new-clock-gate
python /path/to/baselines/phygile/scripts/check_paired_batch_loading.py --manifest /path/to/new-clock-gate/manifest_clock_consistent.json --output /path/to/new-clock-gate/batch_loading.json
```

相对容差 `1e-5` 是处理浮点标头的复现选择，不是论文公布的阈值。标头一致仍不等于外部物理时钟被校准，也不等于动力学跟踪通过。该子集不能替代完整数据；其34条训练序列、6条测试序列、零验证序列仍不足以开展论文规模评价。输出清单引用本机既有文件的绝对路径，移机时应从当地数据重新生成。

其他入口先查看 `--help`。执行环境、检查点与部署源代码路径必须与实际资产匹配。`results/reactive_async_v4/` 和独立复跑记录包含真实调用与校验信息。

新增配对动作诊断的冻结有界命令和范围见 [PAIRED_SCREEN_INVOCATION.md](PAIRED_SCREEN_INVOCATION.md)，对应独立见证和完整批量统计保存在 `results/heartbeat_20261023_0928/`。不能把小样本冻结策略的2/40完成情况解释为论文过滤器或生成模型的性能。

## 完成条件与后续顺序

先完成公开可获得数据与论文算法对应的主线，再将 Unitree 迁移作为补充。PhyGile 仍需全量可获得配对数据、真实物理过滤、课程难度标签、论文网络的两个训练阶段、生成器训练及生成分布上的闭环 PPO。不能将 66 条 SSM 序列或未训练模型替代这些阶段。原文中的私有 3 小时语料不可从公开代码恢复，报告必须保留这一不可消除的差异。

ReactiveBFM 先完成故障后的数值恢复验证，再推进公开代码分支的 400k/1M 里程碑；不得中断其他现有训练。后续增加明确区分的 20/40 窗口对照、指令切换分支、独立评价和实时推理验证。原文表 4 与发布程序存在冲突，按报告保留并核查，不能悄悄选择参数后声称完全一致。

每个新结果须关联检查点、配置、源版本、完整执行日志与失败记录。只在取得新测量后更新指标和视频。v3 备份程序已随故障退出，后续备份应只复制和核验，不重复启动其清理逻辑。本轮不删除数据集、检查点或实验记录。受许可限制的 AMASS/SMPL 原始资产不进入公开 Release。

## 数据与视频可追溯性

`results/public_artifact_manifest.json` 列出发布文件的大小和 SHA256；GitHub Release 的资产摘要须与之匹配。`async_evidence_v1_v4.tar.gz` 保存四次尝试的执行日志、兼容层快照与实际生成/仿真状态，不包含 AMASS 原始训练资产。视频按仿真时间播放，不代表墙钟实时性。

本轮外部辅助筛选的冻结入口见 [PAIR_FILTER_INVOCATION.md](PAIR_FILTER_INVOCATION.md)。`materialize_phygile_auxiliary_filter.py`将筛选结果另存为`pending_phygile_gmt_validation`清单；不得手工改名绕过训练来源审计。全部原始与衍生SSM数组/视频留在授权私有目录，公开结果只含数字、哈希及失败回执。

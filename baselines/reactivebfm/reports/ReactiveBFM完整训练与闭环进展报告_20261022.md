# ReactiveBFM 完整训练与闭环进展报告

冻结观察时间：2026-10-22T01:43:47.778656+08:00。**百万步完整课程训练进行中，尚未训练完成，也未复现论文性能。**

## 1. 实际训练进度与本次交付

原双 RTX A6000 服务器从头训练文本条件运动规划器，两rank均至少完成 **1,220次真实优化器更新 / 1,000,000**，warmup为400,000。不能把两卡更新数相加。训练会话继续运行，后台程序持续备份并进行阶段闭环诊断。

首个1,000更新检查点（官方step999）的模型、EMA与优化器通过有限值/计数检查；模型和优化器、9组闭环证据包及阶段报告均已上传GitHub，并逐件核对服务端大小/SHA256。独立目录中的实际双卡恢复各继续4次更新至1,004后正常退出。

本次训练的是运动规划器；底层跟踪器固定使用官方预训练权重，**没有重新进行跟踪器RL训练**。未发送停止已有任务的命令；没有独立审计它们每个训练步的连续性。

- [本快照训练状态](../results/full_training_20261021/live_snapshot_v4/server_snapshot.json) / [资源与磁盘](../results/full_training_20261021/live_snapshot_v4/resource_snapshot.json)
- [首次模型/优化器GitHub校验](../results/full_training_20261021/first_checkpoint_pair_github_verified.json)
- [完整阶段发布校验](../results/full_training_20261021/first_checkpoint_stage_v1/github_verified_assets.json)
- [原始启动记录](../results/full_training_20261021/full_job_launch.json)

## 2. 数据与训练配置

24个官方公开数据文件合计870,700,765 bytes，本地与服务器SHA256一致；这不是发布者签名校验。175帧窗口过滤后的实际训练条目如下：

| 子集 | 进入训练的条目 |
|---|---:|
| AMASS G1 36D | 1,403 |
| 100style G1 36D | 6,268 |
| Kungfu G1 36D | 687 |
| 合计 | 8,358 |

使用官方支持的逗号组合 `amass_g1_36dim,100style_g1_36dim,kungfu_g1_36dim`，保留各子集发布Mean/Std。默认预合并别名所需的合并目录和全局归一化没有随三份公开数据发布；本方案不声称与其数值分布完全相同。发布Std与简单重算统计也有差异，见[数据审计](../results/data_audit_v1.json)。闭环固定使用Kungfu归一化组件。

| 配置 | 本次设置 |
|---|---|
| 网络 | latent512、16层、8头；107,045,924个可训练参数 |
| 批量 | 每卡512 × 2 = 全局1024 |
| 步数 / warmup | 1,000,000 / 400,000 |
| 优化 | 官方AdamW，lr1e-4，EMA |
| BF16 / 激活重算 | 官方BF16选项 / 自行添加非重入DiT block重算 |
| 种子 | base42；官方rank偏移42/43；这是一个分布式训练种子 |
| cross_prob | 0.0，官方默认 |
| 每rank CUDA分配比例上限 | 0.55 |
| 环境 | 现有Python3.10、Torch2.5.1+cu121兼容环境 |

八卡改双卡并扩大每卡批量以保持全局1024；环境、混合精度、重算和归一化差异仍可能改变数值轨迹。官方226个vendored文件字节不变，补充封装独立保存。运行中train_full.py的SHA256为 `ce21ec77b958132c136eff2e44ebff1940f41eb1f2d6694681ccb864b89d57de`，没有运行中替换。

## 3. 首个检查点验收

| 实际验收 | 结果 | 范围 |
|---|---|---|
| 模型与EMA CPU反序列化 | 各388个状态张量，均有限 | 含buffer的状态元素数不能当可训练参数数 |
| 优化器检查 | 365个状态，step全部为1000，张量均有限 | 检查点完整性诊断 |
| GitHub模型备份 | 857,731,942 bytes，大小及SHA256一致 | model000000999.pt |
| GitHub优化器备份 | 852,468,483 bytes，大小及SHA256一致 | opt000000999.pt |
| 实际GPU恢复 | 两rank从1000继续至1004，exit0 | 独立输出目录；全局1024、原配置与原封装 |

[CPU检查回执](../results/full_training_20261021/first_checkpoint_inspection_v2/receipt.json)、[真实GPU恢复汇总](../results/full_training_20261021/actual_recovery_acceptance_summary.json)、[恢复原始记录](../results/full_training_20261021/recovery_probe_v1/)。

恢复测试后曾因误读启动脚本重复执行一次相同4步检查，第二次也正常退出；[两次原始记录与误判更正](../results/full_training_20261021/recovery_probe_v1_observation_correction/README.md)均保留。这不是第二个独立训练种子，也没有给主训练加8步。直接调用底层训练封装的GPU恢复通过，**恢复助手execute-plan入口尚未经过真实GPU执行验收**。

## 4. 新模型的真实物理闭环

流程：文本生成36D参考 → 官方固定跟踪网络/PD → MuJoCo积分 → 读取实际机器人状态 → 重采样运动前缀 → 再生成。关闭reference forcing，使用real前缀。原生PyTorch适配器复用官方ScaleBFM权重、网络核心、常量与动作适配器；规划器GPU、跟踪器CPU，没有TensorRT。

串行离线仿真中规划暂停物理积分，封装按仿真时间维护反馈历史，并另外记录墙钟耗时。它证明离线物理闭环执行，不能证明实时部署性能。

1,000更新模型完成3条提示词×3个推理种子的9组诊断，每组400控制、1,600次MuJoCo积分、8秒仿真；每组实际状态到后续前缀均通过独立一致性核验。

| 提示词 | 种子 | 最低根高/m | 全程根高≥0.4m | 规划调用次数 |
|---|---:|---:|---|---:|
| perform a punch | 42 | 0.057725 | 否 | 45 |
| perform a punch | 43 | 0.054445 | 否 | 45 |
| perform a punch | 44 | 0.109258 | 否 | 45 |
| stand still | 42 | 0.054691 | 否 | 45 |
| stand still | 43 | 0.054677 | 否 | 45 |
| stand still | 44 | 0.098759 | 否 | 45 |
| walk forward | 42 | 0.054636 | 否 | 45 |
| walk forward | 43 | 0.054671 | 否 | 45 |
| walk forward | 44 | 0.091643 | 否 | 45 |

**高度代理达标为0/9，模型仍会跌倒，当前不能作为有效控制baseline结果。** 这是3条提示词的有限仿真诊断，不是论文成功率、文本语义分数或完整测试集评价；3个推理种子不等于独立训练种子。

- [九组原始回执、反馈核验与日志](../results/full_training_20261021/first_checkpoint_stage_v1/)
- [阶段报告](https://github.com/Pendragon08/conditional-human-motion-generation/releases/download/reactivebfm-full-training-20261022/full_checkpoint_000000999_report.md)
- [完整仿真证据包，含状态、前缀与8秒视频](https://github.com/Pendragon08/conditional-human-motion-generation/releases/download/reactivebfm-full-training-20261022/checkpoint_000000999_evidence.tar.gz)
- 本地视频：`D:/具身智能/ReactiveBFM复现_20261021/ReactiveBFM_1000步_站立闭环_8秒.mp4`

另有同一1,000更新模型的100控制短诊断：12次规划、2秒仿真、墙钟12.574秒，最低根高0.063620m，[回执](../results/full_training_20261021/first_checkpoint_closed_loop_smoke_v1/receipt.json)。它与9组400控制诊断分别记录，不作为第10个独立样本。

前期静态A-pose参考400控制/8秒的最低根高0.788038m；旧101更新模型100控制时跌倒。这些分别是固定参考诊断与旧模型结果，不混入新检查点表现。完整规模文档运行两rank各4步、FP32重算对照365个非零梯度张量最大绝对差0，见[运行见证](../results/full_training_20261021/document_witness_v1/witness.md)。

## 5. 后续自动训练、评估与保存

目标仍为百万次更新，当前尚未进入400k之后的完整课程阶段。实际1,000更新先保存一次，之后每10,000个官方step保存并确保最终保存。step从0计数，文件step999对应1,000更新，step10,000对应10,001更新。

本地后台程序每300秒检查ready清单：下载模型/优化器并核对SHA256 → 上传Release核对大小/摘要 → 阶段闭环与报告/证据上传 → 允许按规则清理已验证备份的旧副本。已实际完成首个完整阶段的这条发布链；**首个检查点仍是最新，未发生本次完整训练旧检查点清理验收**。

阶段诊断在1,000、10,001、每100k附近和最终1M进行相同9组测试；各阶段保存站立seed42视频。缺少回执记录为失败，不推断物理步数。GitHub只追加且不覆盖不同哈希资产；保留全部已发布检查点。服务器保留最新，本地保留最新、1,000更新与100k里程碑。

缓存也核对远端大小/SHA256；回执封印后复用前验证。完成需两个正确rank均达百万次实际更新，且最终检查点/诊断证据完成发布。进程按本次输出目录精确匹配；状态超过30分钟未更新或进程消失会发布异常记录；不会自动终止或重启训练。

原始训练动作、AMASS、SMPL-X和第三方权重保留授权本地/服务器，GitHub保存来源、划分、哈希与汇总，不再分发这些原始包，也未删除它们。模型训练产生的检查点、仿真状态和视频作为本次实验产物发布。

## 6. 审计与恢复边界

独立104393d初审、6850b6e复审、50e5784复审均为same-family/provisional **WARN**，原报告逐字保留。最后一次审计冻结在810更新、尚无ready清单的版本；本报告新增的真实1,000更新CPU/GPU恢复与9组闭环证据没有由该审计重新审阅，不能把原WARN改写为PASS。

数值与证据类型明确区分，没有以预测冒充真值；百万步与论文指标仍未完成。缓存验证、分页、精确rank/进程匹配、停滞记录和删除保护已修正。11项备份协议测试、7项恢复协议测试通过，使用虚拟字节，不是机器人性能实验。

恢复助手默认只生成计划；显式execute-plan在Linux检查原任务已退出，重新核验模型/优化器、清单、args、配置及**train_full.py封装脚本**，然后向新目录启动。这不是启动时对全部上游代码/环境的重审；备份进程不自动调用它。没有随机数/数据迭代器逐位恢复保证。

训练器仍直接向最终文件名写模型/优化器，完整写完计算哈希后才原子发布ready清单；两文件写入不是原子事务。备份只处理ready清单，恢复必须验证成对文件，不能直接选最大文件名。

- [初审](../results/full_training_20261021/integrity_audit_v1/EXPERIMENT_AUDIT.md) / [6850b6e复审](../results/full_training_20261021/integrity_audit_v1/FOLLOWUP_6850B6E.md) / [50e5784复审](../results/full_training_20261021/integrity_audit_v1/FOLLOWUP_50E5784.md)
- [备份协议测试](../results/full_training_20261021/backup_worker_unit_v8_stderr.txt) / [恢复协议测试](../results/full_training_20261021/verified_resume_unit_v4_stderr.txt)
- [操作说明](../docs/full_run_operations.md)

## 7. 运行条件与尚未完成部分

**本地备份电脑须保持开机联网，SSH/GitHub授权有效。** 程序请求阻止空闲睡眠，未独立验证Windows请求结果；手动关机/休眠仍会中断。服务器训练独立运行，空间不足时等待保存，不删除未备份文件；长时间等待可能触发默认3小时分布式超时。训练/备份退出后不自动恢复，须查看记录并核验检查点。

本快照1,220次更新耗时5355.7秒，约4.39秒/步；按早期恒定吞吐外推剩余约50.7天。课程阶段、共享负载、检查点上传和评估都会影响耗时，这不是完成时间承诺。每rank训练张量峰值约10.7GiB，整卡使用还含已有任务；本次资源记录给出磁盘现状，后续依赖验证备份后清理旧副本维持空间。

尚未完成：百万步完整训练、后期稳定动作效果、论文完整性能对比、实际旧检查点清理验收、execute-plan GPU执行验收。目标/到达数据与完整官方评测不可用，TensorRT实时性能未测，没有真机结果。

- 服务器：`/home/ouyanghao/reactivebfm-full-20261021/training_v1`
- 本地：`D:/具身智能/ReactiveBFM复现_20261021/full_training_20261022`
- 本地自动阶段报告：上述目录的`最新训练阶段报告.md`
- [GitHub训练产物与阶段报告](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/reactivebfm-full-training-20261022)

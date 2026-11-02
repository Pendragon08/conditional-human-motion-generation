# 用户100k模型：配对生成诊断与生成动作物理验收

最新实际物理采样：2026-10-29T02:01:37.804770+08:00。VAE与LDM用户指定100000步已结束，继续使用其不可变状态，不追加训练。

32个原val来源、188个独立标注段/块、5个固定seed、同噪声文本条件CFG5/无条件配对，共940组/1880次生成，全部真实完成，0失败、0未尝试。生成运行约1510.28秒。完整journal/每个NPZ字节核验通过；随后独立CPU作业4.17秒复算9400个真实数组的SHA/CRC/shape/float32/finite和Huber。CPU模型加载/GPU/优化更新均0。

| 归一化Huber（越低越接近该段参考） | 文本条件CFG5 | 无条件 |
|---|---:|---:|
| 每trial等权均值 | 0.29119397 | 0.21918472 |
| 按计分帧加权均值 | 0.34884011 | 0.24721529 |
| 每来源family等权均值 | 0.32954295 | 0.25621181 |

条件误差在940配对中313组更低、627组更高，均值比无条件高32.85%。全部5seed条件均值均更高。保留此负面结果；它衡量参考重构，不能据此声称TMR/文本语义改善或退化的正式评价结论。

原始contact未经clamp：条件129016/195580个值在[0,1]外，范围[-0.12028384,1.02303267]；无条件114559/195580，范围[-0.03429115,1.01399958]。这是连续输出的描述，未据此篡改生成值或记成物理失败。最大root quaternion平方范数偏差3.76e-7。

实际Isaac articulation FK已对全部1880条/195580帧完成，29关节按真实名称映射到Isaac29关节/30body，joint_pos对写入状态逐批吻合。0动力学、0神经模型、0优化更新。私有FK NPZ共424260880B，完整manifest/全部输出SHA与1880真实输入覆盖核验通过，process/supervisor/outer0。原生SDK close未返回，保留native_close_returned=false、teardown_fallback_used=true及真实闭合0；这不是原生teardown成功。

新生成动作严格回放：USER_LDM100K_GENERATED_STRICT_PHYSICS_FINISHED；结束1880/1880，通过1873，失败7，实际闭合True。按时长排队以减少padding，所有seed/mode全保留；进行中计数不能当最终通过率。冻结15000轮策略，原8278训练来源fingerprint仍独立保留，新生成library fingerprint为f2ec3f7f685071003c349c04dff8930df328bf988c28d0f46236be70121ecb1d。没有恢复teacher训练或改变其权重。

物理沿用已实跑的strict T-1覆盖和失败优先、557D actor、29关节/14body、16环境、50Hz控制/200Hz物理，pelvis高度0.25m、端点高度0.5m及原orientation终止不变；solidfloor4000×4000×0.2/topz0/friction1/restitution0/multiply/selfcollision不变。SDK/模型共同锁串行。新增独立终点位置距离使用初始pelvis yaw/xy固定对齐，比较实际终止前捕获状态与最终参考帧；提前失败也保留其距离，并独立标记T-1是否完成。此距离不会改变原strict通过规则。真实执行姿态和所有失败始终私有。

本次可修复入口失败全部保留：CPU summary_v1把合法SHM清单误按DATA根检查，真实child/sup/outer1、无模型/GPU，归档原字节后summary_v2只选择明确ROOT/SHM允许根，并将新增私有输出放已实测HOME；未降低任何owner/no-symlink或3GiB DATA保护。FK_v1实际SDK构造/FK首条与一个孤立输出已发生，因预算遗漏原角速度float64晋升而在字节上界处退出1（0已提交trial、空manifest），原输出/失败/source完整保留。核真实退出/无worker/锁空后，FK_v2仅把输出预算由1792修正为2152B/frame（358float32+90float64），新容量接受后才运行；没有改变角速度数值、clamp、数据、模型或阈值。单条未提交FK重新执行，未重跑1880次LDM生成。源归档见sources/ldm100k_summary_failed_v1与sources/ldm100k_fk_failed_v1。

当前仍非全1774family生成、连续转折、正式TMR、WBC在线或完整论文效果。缺少可验证机器人69D兼容的文本运动特征器，不能用CLIP、Huber、随机或任意人体特征冒充Rprecision/MMdist/FID。真实物理闭合后自动汇总全部mode/seed/失败与终点，再做实际执行影片和后续WBC/语义验收，不询问是否继续。

CPU summary receipt SHA 6da99482e8f46c45f6c4755758080178e9eb18a74dbb8b470426026b75bed806；FK receipt SHA e2bb1ccd5c2cb795ee54aeb21dcaf2a11d83d022da68ea2ad56f71a05085a768。完整模型、数组、逐源文本/轨迹和私有SDK不公开，公开只代码与聚合证明。所有fs/D至少2GiB保留，每新job实查；HOME/data当前采样不能当预留。原源/spec/失败/partial/旧checkpoint不删除，不重跑旧清理/审计/夹具。

## 全部新生成动作的真实物理终态

| mode | 完整T-1通过 | 失败 | 通过比例（本次单次回放） | 最终参考14body平均位置距离，全部trial |
|---|---:|---:|---:|---:|
| conditional_cfg5 | 934/940 | 6 | 99.3617% | 0.097606m |
| unconditional | 939/940 | 1 | 99.8936% | 0.073061m |

全部1880次结束：1873完整通过、7失败、0未尝试，process/supervisor/outer0/closed/无worker/锁释放。独立审计195105个有限执行姿态样本、28095120B真实qpos与全部journal/trace SHA及失败优先T-1规则通过。实际12291次控制/49164次物理；终止原因5次orientation、2次末端高度。配对统计{"both_complete": 934, "conditional_only_complete": 0, "unconditional_only_complete": 5, "neither_complete": 1}。原生SDK关闭未返回，20秒有界退出事实保留。终点距离与strict通过规则分别报告，不把99.6%的本次单次通过比例当作失败概率证明或正式在线质量。CPU独立核验不启动新模型/GPU/SDK。physics receipt SHA b110c1cd51babe40a6cfab758525fc451a54e366e01706a355ed132d00d017b8。

## 全部真实影片与本地接受

CURRENT最终覆盖：9/9真实执行影片已全部CPU/Mesa完成，batch/全部clip监督/outer真实0、无worker/commonlock释放；全部7失败与固定首配对完整保留。9片共3496454B，现已全部D媒体/PNG完整SHA及Windows当前owner核验，D/media/user100k_effects_20261029；D/TextOp十万步实际效果_20261029.html可直接查看条件/无条件与全部失败实录。原数组/逐源文本/fullAdam未下载到公开位置，MP4本轮只D私有展示，未发公开Git。停止对generation/summary/FK/physics/影片的重复运行或已完成审计，后继继续真实兼容语义/TMR/WBC/在线质量；不因本次物理99.6%而宣称论文完整复现、不追加生成器/teacher训练。现无活动模型/SDK/影片job，只有独立私有LDM字节副本pld100_v4仍待实际核。

服务器batch receipt SHA `c27589540db155e7e7ec3acf04e15625cda24c195ea119aca837fecf8c0cec45`；全部媒体账本见results/ldm100k_user_handoff_20261028/user_ldm100k_execution_videos_v1/local_media_verified.json。只渲染真实捕获姿态，不执行新动力学；首配对与全部失败均未挑选或替换。


<!--tracking-semantic-report-start-->

## 保存执行轨迹的独立位置跟踪核验

新作业user_ldm100k_tracking_fidelity_v1真正闭合0，完整1880条、195105个已保存姿态；约39.60秒，0新动力学、0策略回放、0神经模型、0优化更新。对原Isaac articulation按真实关节名进行FK，不重新生成动作或物理执行。独立collector核验全部private标量记录与原物理覆盖/失败决定join、全部SHA、门槛和聚合均通过。

[论文附录B-B](https://arxiv.org/html/2602.07439v1)的Succ公式使用逐帧平均根相对关节位置误差的时间最大值≤0.3m。这里补充要求实际完整T-1覆盖，提前失败只有真实已执行前缀，绝不补造缺失帧。原终止规则完整数934/940与939/940另行保留；新条件多拒绝1条，不篡改原严格物理决定。初始样本排除，已执行states1..steps比较真正被消费的reference0..steps-1，固定初始pelvis yaw/xy对齐并保留参考初始z。参考终点距离与原独立捕获结果最大偏差1.18e-6m。

| mode | tracked14完整覆盖且≤0.3m | 每trial等权global MPJPE/mm | 每trial等权root-relative/mm | 按实际观察帧加权global/mm | 按帧加权root-relative/mm |
|---|---:|---:|---:|---:|---:|
| 文本条件CFG5 | 933/940 | 70.414105 | 35.092239 | 87.207492 | 38.361214 |
| 无条件 | 939/940 | 57.640415 | 30.710351 | 64.100542 | 31.579321 |

tracked14包含pelvis；另提供全部30个Isaac link：完整覆盖且≤0.3m仍933/940和939/940，逐trial等权global为70.087456/57.357691mm，root-relative33.665078/29.466485mm。原论文J的精确资产集合未获得证明，故这是明确body集合的补充核验，不宣称与论文完全同协议。所有trial包括提前失败均保留观察前缀误差；不能用仅前缀低误差的936/940与940/940冒充完整成功数。32family的单次短段结果不是全val、失败概率证明或在线质量。原native_close_returned=false/fallback=true保留。

完整跟踪收据SHA 77ce67f28800ec4dd9f25cfb52fe12c89d99d1e7d7f1af2c9b34ab9f7ad8d167；新可复算private标量与执行姿态只server+D私有，公开仅聚合证明。没有改CFG、阈值、标签、原始contact或生成器权重。

## 真实语义评价器的后继进度

TMR独立CPU环境tmr_isolated_runtime_v1真实0/closed，6个固定SHA公共wheel私有--target安装与原Torch源不变核验通过，3个CPU随机架构构造/varlength有限输出，16031301参数/64125204参数字节/201326592B保守Adam状态上界；不是正式初始化、不是已训练评价器。唯一后继tmr_text_models_v1 status=FAILED_TMR_TEXT_MODEL_PREPARATION_ALL_PARTIALS_RETAINED，closed=False，当前commonlock_free=True；只准备固定revision的DistilBERT和MPNet公共文本模型707323471B及冻结CPU检查，0机器人数组/BABEL文本/评价器训练。实际collector决定新状态，不能把派发/下载或随机架构冒充正式TMR/Rprecision/MMdist/FID。后继继续真实TRAIN-only69D数据绑定、文本缓存、完整梯度/恢复与独立验证后才可训练评价器；不恢复/延长VAE/LDM/teacher。当前无新生成器、物理回放或影片作业，全部9片D已核验，不重跑或重复下载。
服务器直连公共HF在270.79秒后真实Network is unreachable退出1，process/supervisor/outer1、无workers/锁释放、0完整模型文件；失败收据与0B partial保留，禁止重派该run。替代传输public_text_cache_v1只在本机现有代理下载同revision/同SHA公共文件至D/models/evaluation_text_tmr_20261029，status=DOWNLOADING_PINNED_PUBLIC_TMR_TEXT_FILES，完成1/11文件；其下载不加载模型或占GPU锁，原LDM私有copy独立继续。必须先真正完整bytes/SHA/nativeowner/fsync/rename/原进程退出，才可签名上传与服务器冻结CPU核验；缓存不是已训练TMR。工具exec会话61659仅线索，实际PID/receipt/resources/exit为准，不盲重派或删除partial。


固定原始MIT源：[Mathux/TMR](https://github.com/Mathux/TMR/tree/6d74688730d15d43b0a755ce2b0e1f2d76138fc1)，commit6d74688730d15d43b0a755ce2b0e1f2d76138fc1/treecbb99c6b2cc724b2d105795ce6b696e7ba31af6b，原Git blob/SHA精确保存在sources/tmr_upstream_20261029/provenance.json及LICENSE。官方6层ACTOR架构只将motion输入维度显式适配为真实69D，正式训练入口尚未接受。原TMR源码标明500epoch、Adam1e-4、重构/latent/KL/InfoNCE损失；未来训练方案必须签名绑定，不把源码默认当已执行。

发现[Motius公开G1 TMR](https://github.com/ZeyuLing/Motius/blob/main/docs/evaluator_zoo/g1_tmr.md)是真实机器人候选，但其38D/30Hz、HYMotion训练集与本任务69D/50Hz/BABEL来源不同，且其文档对FID做逐样本L2归一化；尚未验证训练family重叠/协议一致性，未下载其权重或拿来报告本任务TMR。不能用CLIP/Huber/随机特征/任意人体编码器充当Rprecision/MMdist/FID。

后继只用TRAIN family，保留原val1774family用于最终评价，不对生成的32family结果调CFG或选最好seed。数据、规范化、文本缓存、容量与完整Adam/RNG恢复和独立验证必须真实完成。必要评价器学习不等于给生成器或teacher追加训练；可部署431D actor的WBC与在线质量仍需单独真实验收。
<!--tracking-semantic-report-end-->

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

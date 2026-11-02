# 十万步生成动作：独立跟踪精度与语义后继


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

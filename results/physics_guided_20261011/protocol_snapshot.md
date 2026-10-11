# 物理引导条件动作生成：执行协议

本轮固定生成模型与控制器，研究参考动作构建；人体是CLoSD的SMPL模拟角色，不是真实机器人。数据来自作者缓存的HumanML3D。按官方`index.csv`的原始MoCap source_path分组，镜像、不同时间切片不跨组。使用本轮独立分组，不能声称排除了发布权重的预训练暴露。

已有运行环境配置为`configs/env-closd-baseline-20261010.json`，规范哈希81bdf764e312bcc92e4969e5eb9d2d6d6043f14c037e5a588dabbd564871f780。包和上游源码不改动。实验设置另见`configs/physics_guided_20261011.json`，不能混同软件环境规格。

先验判据包括固定目标位置10cm、朝向10度、目标时刻0.1秒、跟踪均值10cm/p95 20cm。足部滑移是碰撞表面几何+刚体接触力代理，不是实测接触点。腾空不作为漂浮判定。验证集上校准后冻结判据，再运行测试；不得用测试结果修改判据。

生成模型支持终点位置和朝向，不支持任意接触力或音乐。先播放40帧人体前缀（20Hz，即2秒），DiP仅接收其中最后20帧（1秒）作为生成上下文，再预测40帧（2秒）。因此运行时打印context_len=20、pred_len=40，与4秒总动作长度一致。物理控制30Hz。测试目标取数据集终点，按世界坐标水平偏移±15cm。目标先于生成固定，物理精化后仍用同一目标。

独立文档执行见证：仅执行以下指令，不修改、修复或另起正式队列。保留失败输出。

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 'bash /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/physics_guided_run.sh check witness_doc_v4'
```

预期零退出；依次打印PHYSICS_GUIDED_CORE_WITNESS_PASS、PHYSICS_GUIDED_ANALYTIC_SURFACE_CHECK_PASS、CLOSD_R3_GPU_WITNESS、PHYSICS_GUIDED_COLLECTION_PASS。`results/physics_guided_20261011/witness_doc_v4/receipt.json`应有6组首轮尝试；失败与提前终止也保留在分母。joint_pass不要求为6。同目录`metrics.json`是6个逐尝试记录，含case_id、length、completed和terminated；`trajectories.npz`含valid掩码和terminated数组，`metadata.json`记录来源及目标。reference_frame必须等于0至119且不重复首帧；原生模块在观测后增加frame_idx，适配器统一使用独立时钟，实际前缀也补入当前刷新状态。receipt只作汇总，不替代逐尝试证据。原始数据不上传GitHub。

正式队列：通过见证后主执行者启动`bash scripts/physics_guided_run.sh queue`。上游仓库保持clean；新机制在实验包装层实现，不能称为原版CLoSD完整论文复现。新机制固定原始目标、最多3轮，有界位置/朝向调整，使用与随机候选选择相同计算预算。

主结果须使用独立回放：原始参考与精化参考均提交固定策略重新执行。原始生成执行得分不代替精化数据的独立回放表现。增加有限次数反馈、随机候选、打乱反馈、无反馈以及仅按跟踪指标选择的消融。人体门槛不满足时停止G1及下游训练；不给失败路线自动换指标、增加候选或调用付费VLM。

候选方法默认使用实际执行前缀和实际坐标锚点；计划前缀消融同时替换两者。打乱反馈按整个原始来源组置换，保留同一目标偏移，不能仅在同来源的三个偏移之间轮换。语义与多样性采用缓存的HumanML3D文本-动作评价模型，是学习型代理而非人工真值；所有失败轨迹仍保留，不能靠去掉失败来提高分数。

协议偏差须保留：本轮数值阈值在验证集运行前即写入文件，验证主要用于坐标、时间、接触测量和分布检查，不是依据验证分布估计新阈值。测试后不会放宽阈值。若验证集联合合格率为零，应报告可能存在门槛地板效应，不能将零提升解释为该类方法普遍无效。

记录总GPU占用时间（单GPU，按子任务GPU进程墙钟时间保守计费）≤72小时，包含失败任务。各阶段上限4/12/12/36小时及8小时故障预留。新输出≤15GiB、数据盘剩余≥20GiB。除本轮进程外不终止任何进程，不删除旧项目。下游训练与G1只有已审计适配接口并通过人体门槛才可运行；接口或语义验证缺失时不能宣称完成。

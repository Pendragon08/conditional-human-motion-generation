# TextOp快速复现：真实跟踪评估与录像

更新：2026-10-26T19:27:43+08:00

**当前属于5,000轮策略的快速验证；论文40,000轮完整版、全库物理筛选和文本生成—执行闭环尚未完成。**

|阶段|实际状态|
|---|---|
|可得数据预处理|8,278条原始动作＋对应镜像，GMR、文本时间对齐、69D编码、FK和私有持久备份已完成|
|可恢复跟踪策略|5,000轮／100,000次优化更新完整检查点，服务器和本地私有备份SHA一致|
|受控结束|连接恢复后完成备份再发送一次SIGINT；原监督器退出0，原始最终收据保留|
|断连造成的额外训练|最终观察到5400轮／108000次更新；5,000轮之后的内存更新不保证可恢复，实际评估使用5,000轮快照|
|数据与原始记录|数据集、检查点和失败记录全部保留；旧57D、PhyGile、ReactiveBFM继续暂停|

SSH于本轮检查恢复，没有修改代理、路由或ZeroTier设置，先前断连根因仍未确定。原失败协调器coordinator_v1和新的安全接续coordinator_network_resume_v2均保留，未重启训练器或改动其16份活动文件与3份控制源码。

## 已完成的实际动作回放

|运行|动作数|自然完成|失败|向量控制调用|物理子步|真实退出|
|---|---:|---:|---:|---:|---:|---:|
|quick_pilot_sample64_v1|64|42|22|4650|18600|0|
|quick_pilot_smoke_author_v1|8|8|0|63|252|0|
|quick_pilot_smoke_document_v1|8|8|0|63|252|0|

两次smoke是对同一组8条完整短动作的独立重复验证，不能相加称为16条不同来源。代表性sample64另选跨10个数据集的64条完整原始动作，与smoke选择不重叠；未尝试来源不能判为失败或物理不可行。

## 自定义跟踪误差

下表按每条试验实际控制步加权，包含失败前已执行的步数。它比较自动重置前的真实机器人姿态与当前消费的控制目标，不是论文基准分数，也没有独立验收最后参考帧端点。

|运行／统计范围|试验数|实际累计试验控制步|14身体位置平均误差 m|29关节RMSE rad|
|---|---:|---:|---:|---:|
|quick_pilot_sample64_v1 / all|64|23110|0.06256|0.23918|
|quick_pilot_sample64_v1 / nominal_duration_completed|42|19818|0.05640|0.20360|
|quick_pilot_sample64_v1 / failed|22|3292|0.09966|0.38992|
|quick_pilot_smoke_author_v1 / all|8|450|0.06823|0.19274|
|quick_pilot_smoke_document_v1 / all|8|450|0.06823|0.19274|

SDK原生关闭仍未返回，实际使用了有界退出fallback；退出0不代表修复了原生关闭。训练和回放沿用同一静态箱体地面、自碰撞与严格阈值；作者精确碰撞资产不可得，仍是复现差异。实际CUDA/Torch在GPU1、CPU0–3、Torch显存比例0.65；SDK在GPU0的辅助上下文按原资源记录披露。环境seed沿用既有配置20261025，执行脚本Torch seed为20261026，不能把两个设置混为同一个随机源。

## 实际录像

- actual_sample64_longest_completed_v2：1613帧、50fps，约32.26秒，7000764字节；媒体SHA 28f75b8210af35810b3ce4ab63018bfbfb4cb7de5157c32683c176e53e1df169。
- actual_sample64_longest_failed_v2：440帧、50fps，约8.80秒，1654976字节；媒体SHA 24035899475a575860b4b524d86d4b63a5aaba979e21e1e36a4e752a9d696bd9。
- actual_smoke_author_trial0_v1：52帧、50fps，约1.04秒，341061字节；媒体SHA 2d9a9a1f9ae272e2f830fbe82dd6d0cbb3cf8e166e97bcfa634faf17b9949129。
  [GitHub科研结果视频与署名说明](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/textop-quick-tracking-video-20261026)。

录像是已记录IsaacLab物理执行姿态的逐帧可视化，由CPU/Mesa和MuJoCo视觉FK绘制，新增动力学积分和优化均为0。它不是参考动作动画，也不是MuJoCo跨仿真控制或生成器闭环视频。视觉MJCF与跟踪器URDF来源不同，不能据画面声称碰撞几何相同。原始轨迹和逐条标注保存在服务器与本地私有目录。

SFU来源的成功示例以非商业科研结果录像发布，并附数据来源与资助致谢；[SFU官网条款](https://download.is.tue.mpg.de/amass/licenses2024/SFU/sfu_license_updated.html)说明科研使用及发表结果时的致谢要求。BMLmovi来源的失败录像保留本地；其[官网许可](https://download.is.tue.mpg.de/amass/licenses2024/BMLmovi/bmlmovi_license_updated.html)对第三方分发的限制更严，不随公开代码发布。没有公开原始动作数组、人体模型或逐条文本索引。

长片渲染记录：32.26秒完整轨迹的首次CPU/Mesa导出在900秒监督时限退出，实际退出−15，保留了不完整MP4和全部日志；它不是有效结果视频，物理评估结果与闭合轨迹没有丢失。根据52帧短片36.76秒的实测速度，仅将完整导出的监督预算改为2,400秒，帧率、分辨率、所有输入帧和渲染数值签名不变，新版本使用独立run并保留实际退出证据。更长的成功与失败录像以最终收据为准，不能把后台启动或部分文件算成完成。

## 下一步与范围

64条代表动作已真实评估：自然完成42、失败22。这些是早期策略的本次试验结果，不能作为动作绝对物理可行性的结论。
来源选择保留42条名义跟踪完成的原始参考，train/val来源为{'train': 31, 'val': 11}；未评估镜像排除。来源view的历史training_ready=false与后续独立语料/训练入口验收分别记录。
快速版私有参考语料与缓存已完成独立验收：42条原始参考、31条训练来源与11条验证来源、174个动作—文本标注对、74个非空CLIP文本，训练归一化统计包含14,505个69D特征帧。来源和精确float32姿态跨划分重叠为0，未加入未评估镜像；TRAIN/VAL/weightedTRAIN三项实际512批次、四primitive加载均通过，history连续误差0、缺失文本嵌入0。全部私有语料已在服务器与D盘核验备份。

## 快速VAE实际训练

服务器运行quick_vae_train_v1，最近核验状态QUICK_VAE_2000_TRAINING_FINISHED：2000/2,000次primitive训练尝试、2000次实际Adam更新、0次跳过。VAE使用69D、batch512、history2/future8、四primitive和原网络/损失；课程及学习率时钟压缩100倍，属于快速版，不等同论文200,000步。

实际子进程退出0、监督验收True；固定验证批次的VAE重建门槛True。这些是后验重建诊断，不是文本生成质量或物理执行指标。
已实际写入的完整状态步数：[1000, 2000]。每份保存后重新加载，核对模型、优化器、计数和所有随机数状态；不删除已有版本。

LDM3,000步使用独立入口和真实已训练VAE验收，不以梯度诊断权重代替。原论文VAE200,000/LDM300,000配置保留；当前文本生成—执行闭环仍未完成。

第一阶段5k特权teacher原型不能冒充论文最终可部署WBC或完整实时系统。

数据范围差异：13条所选CMU导出缺失，403条作者私有动作不可得；8,278条来源不等于作者全部12,296条公共跟踪动作。公开仓库保存代码、聚合证据、推理权重、许可允许的科研结果录像和报告；完整Adam/逐动作采样状态与受限资产保持私有。

[5,000轮推理权重与说明](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/textop-filter-policy29d-20261026)。完整恢复仍须私有检查点。

证据目录：results/quick_pilot_actual_20261026、results/formal_filter_checkpoints_20261026/round_5000及results/quick_reproduction_20261026/coordinator_network_resume_v2。
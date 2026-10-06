# Isaac Lab 优先：执行反馈与世界模型辅助 PPO 新阶段

用户本轮要求继续证明执行反馈、设计世界模型与PPO联合优化，并恢复正确平台顺序。该协议继承旧试验的否定结果，不覆盖它们。由 `feedback_world_ablation_review_v2_20261006` 独立先给出消融设计后落地；新方法尚未验证。

## 平台及证据顺序

原 DreamControl 的11项控制策略已经在Isaac Lab训练。上轮MuJoCo冻结策略试验用于便于复制物理状态的模块筛查；它不能替代Isaac Lab方法主实验。新顺序为：Isaac域内数据/表示与动力学门 → Isaac反馈或PPO消融 → 冻结相同checkpoint/接口 → MuJoCo zero-shot sim-to-sim。MuJoCo不用于主方法调参，不只保留迁移成功臂。旧learned-MPC控制FAIL完整保留。

## F：真实执行反馈进入生成器

PPO已有本体状态输入不是新增F。F把已执行的偏差送入冻结OmniControl，重生成未来人体参考，经原PyRoki和官方时序整理后，让G1在**当前真实物理状态**继续跟踪。不增加额外机器人IK，不重新设置机器人到新轨迹，不移动原世界目标，不重启动作时钟。

先进行现有9样本原重定向/Isaac继续执行预检：这是nominal MuJoCo反馈、单扩散seed和已暴露的3参考，只能检查接线和失败机制。其后在Isaac采集80N/0.2s横向扰动后的实际状态，设计F0固定参考、F1同预算planned-context重生成、F2简单root/yaw重对齐、F3真实root/踝部稀疏残差、F4同任务同相位循环错配残差。真实反馈仅使用branch之前状态；3个扩散和扰动seed。

原生成每批约100s，初始试验是同步暂停物理时钟的离线pause-and-replan。所有重生成臂相同预算、相同等待；报告墙钟。物理改善与实时性是两个门，前者通过也不能声称在线闭环。完整263D执行前缀尚未建立，稀疏反馈不能称完整CLoSD移植。

正式F贡献门：重定向接受/有限/限位100%；正确F恢复AUC相对F1与F2均改善≥10%，按reference配对bootstrap两CI下界>0；优于F4；任务成功下降≤5pp、跌倒不增、clean跟踪非劣5%、切换尖峰P95恶化≤10%。新确认参考必须未参与旧桥拟合/测试；现有0–65均已经暴露，阶段试验不得称最终未见动作泛化。任一门失败先定位或停止该方案扩展，不将失败样本从分母删除。

## W+PPO 首选路线：动作条件的预测表征辅助

借鉴ICRA2025 WMP的世界模型表征＋真实仿真PPO，针对G1参考运动跟踪提出新适配。WMP原任务是A1视觉地形行走，不能直接声称其证明了人形运动跟踪。单纯组合已有模块也不是已经成立的创新。

接口：历史本体观测和已执行动作 → 小型GRU潜在状态h → 动作条件世界模型预测未来1/5/10控制步的关节变化、身体姿态/速度、支撑滑移及接触标签 → 将h作为PPO actor/critic额外输入。actor只能用已有可部署观测；真实未来物理状态和接触力可作为监督标签，不能作为actor当前特征。原任务未来reference是计划输入，不等于实际未来状态。

可先使用128维h、共享历史窗口约0.2–0.5s；具体容量/损失权重只在Isaac validation选择后锁定。初期保持原任务奖励、PD、动作范围、episode、参考和扰动分布。奖励、done、GAE advantage和return全部来自真实Isaac rollout。正式更新期世界模型用同一批真实状态转移训练，不为某一臂额外收集数据。域内预测门使用所有臂共享且单独计账的Isaac warm-up数据，来源及预算见下文。接触监督是实际地面传感量，Kick无实体目标时没有打击接触监督。

更新循环（以版本冻结和stop-gradient保证PPO比率一致）：

1. 冻结世界模型版本φ，在当前PPO版本θ下采集真实rollout；缓存每步h、动作、old log-prob、真实reward/done/value。
2. 从真实reward和value计算GAE与return。PPO全部epoch使用缓存的相同h和old log-prob，世界模型参数不在这些epoch中变化。
3. 用原PPO clipping、value和entropy目标更新actor/critic。actor输入`[o, stop_gradient(h)]`，不把PPO梯度写进世界模型。
4. PPO周期之后，独立更新动作条件世界模型；多步损失窗口不得跨reset/done。标签及归一化来自训练分组。
5. 下一个rollout开始时以过去已观测的短历史重新计算新版本h，reset时清空历史；不能把旧参数生成的隐藏状态当新模型状态。

先采用独立表征路线，避免共享encoder同时辅助更新与PPO epoch内特征漂移。共享encoder的联合辅助loss属于后续独立消融，不混入主方法。预测风险或可微动作正则仅在表征门通过后增加；其系数、收益及原始未整形回报单独记录。

实现不变量：`h_t`只能读取`o_≤t,a_<t`，不包含尚未采样的当前动作；单步预测为`f(h_t,a_t)`，H5/H10辅助预测使用已采集真实动作序列`a_t:t+k-1`，这些未来动作不进入当前actor的h；世界模型输入归一化器也在整个rollout/PPO周期冻结。rollout buffer缓存h、wm_version_id、世界模型输入归一化版本、actor观测归一化版本、feature schema hash、hidden reset mask和episode boundary；末端bootstrap value也用本周期同一模型／归一化版本计算。PPO minibatch不得重新计算乱序GRU历史。GAE区分真正termination与time-limit truncation，timeout bootstrap严格复用固定官方RSL-RL语义，避免重复或漏算；世界模型窗口对任何reset都断开。正式训练恢复原task的全部termination，不能继承本轮单回合诊断禁用termination。真实未来仅作标签；critic privileged输入与actor latent隔离。归一化、W replay和checkpoint选择不使用validation/test reference。state-only保持相同通道/参数，仅将动作通道置零；错误监督在同任务/相位/接触层内、跨不同episode/reference打乱以保留边际分布，不允许相邻时间窗口泄漏。

这与上轮“冻结PPO，再用模型挑动作”的MPC路线不同：这里PPO权重从真实物理rollout重新学习，世界模型参与训练/推理特征。上轮MPC失败不证明新路线一定失败，也不为新路线提供有效性证明。

## W+PPO 消融与匹配预算

|臂|作用|必须固定|
|---|---|---|
|P0 原PPO等预算续训|控制额外RL训练本身|同checkpoint、参考、真实步数、PPO更新|
|P-cap 容量匹配PPO|排除网络变大|总可训练参数、输入信息、主优化预算|
|P-history 同历史GRU的PPO|排除历史记忆而非预测的收益|历史窗、h维、actor输入容量；GRU只靠PPO训练|
|W-correct 动作条件多步世界模型特征|主方法|真实rollout步数、初始化、PPOepoch/minibatch|
|W-shuffled 同架构错误转移监督|测物理预测内容因果贡献|目标幅度/采样、模块容量和辅助更新数|
|W-state-only / one-step|测动作条件/多步作用|单独扩展，不事后替换主臂|

新增输入层的h权重初始为零，尽可能保持warm-start actor初始函数；核对相同初始动作后才训练。各臂共享训练seed42/43/44、真实环境步数、扰动、学习率/clip/epoch/minibatch/PD/参考，单列世界模型优化计算及训练/推断墙钟，不能将同环境步数写成同总计算量。

先Kick，初始建议各8192环境×24步rollout×500PPO更新作为预注册匹配预算（原2000轮的25%，并非严格论文全预算）。每臂每seed98,304,000真实transition；五臂×三seed合计1,474,560,000 transition。更新期世界模型从这些transition抽取监督窗口，不能增加某一臂的真实采样；额外模型梯度步和GPU墙钟另报。预算尚未启动，先做吞吐诊断再给出墙钟估计，不能把transition数量误认为已执行。预测门数据预注册为每seed从共同原始冻结策略采集20个rollout×8192环境×24步＝3,932,160条Isaac transition；按完整reference/episode划分开发训练／validation，且不与最终确认reference相交。该数据只采集一次，所有五臂获得相同warm-up记录，不更新任何PPO；正确／错配世界模型分别监督拟合，历史／容量控制按固定协议初始化，记录所有辅助计算。每臂数据暴露预算合计102,236,160条（共同warm-up＋主训练），按五臂三seed计1,533,542,400条暴露；共享warm-up去重后的物理采集预算为1,486,356,480条，不得把重复暴露当重复真实采样。当前500×18诊断数据只作接线检查，不充足以代替此预测门数据。预测门失败不启动PPO，门的先决成本独立报告。先做域内模型预测门，再三seed因果训练；Kick主门通过后才Punch复核。旧checkpoint已经见过原参考，不能用分组拟合称未见参考策略泛化；最终新增锁定参考用于确认。不预先启动11任务重训。

主门：W-correct恢复AUC比P0、P-cap和P-history改善≥10%、reference配对CI下界>0；优于W-shuffled；成功损失≤5pp、跌倒不增、clean跟踪非劣5%；三个PPOseed方向一致；同时报告固定环境步数下训练曲线AUC/达到预设成功率的步数。预选primary后不因结果改变主臂。Isaac域内预测先要求H10 FK误差比最佳ridge和state-only改善≥10%且CI下界>0；不能直接拿旧MuJoCo predictor做域内证明。

暂不将模型虚构rollout混入PPO。MBPO的短模型rollout原来配合off-policy SAC，不能直接移植为PPO on-policy advantage。若以后研究想象rollout，需要独立方法、真实reward/done校准门和模型偏差对照。

## 启动边界与论文来源

F接口/冻结策略推理现在可执行。新正式W+PPO须先将当前Kick实际仿真状态送到独立本地画面并验收，写入本轮gate；旧Open-Door viewer或checkpoint回放不算。显示故障不能影响计算。准备代码和协议并不代表已训练。

来源：
- WMP作者项目主页，标注ICRA2025：[World Model-based Perception](https://wmp-loco.github.io/)。
- [WMP原文§IV-A/B/C](https://arxiv.org/html/2409.16784v1)：RSSM、stop-gradient潜在状态、PPO真实Isaac Gym数据；明确未使用模型生成rollout训练策略。
- [MBPO，NeurIPS2019](https://papers.nips.cc/paper_files/paper/2019/file/5faf461eff3099671ad63c6f3f094f7f-Paper.pdf)：短模型rollout与偏差分析，属于off-policy路线。
- [CLoSD原文，ICLR2025](https://arxiv.org/html/2410.03441v1)：执行人体状态反馈给下一段DiP；该概念已有先例。

本协议是借鉴已有方法形成可检验的G1跟踪研究假设。只有因果消融、物理门与跨参考/任务确认通过后，才讨论本文具体贡献，不能凭迁移CLoSD或添加世界模型宣布新颖性成立。

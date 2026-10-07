# 残差世界模型＋PPO控制消融：独立审查与实施边界

当前为准备方案，正式PPO未启动。独立 reviewer `/root/world_residual_ppo_ablation_review_20261007` 按 ablation-planner 阅读当前代码、协议和结果提出下列要求，执行者再核对原RSL语义。新11参考单次预测确认三seed/clean/recovery必须先PASS；程序接口阻塞不当方法FAIL，科学FAIL则停止当前候选，不改门重测。

## 主比较与必要解释

原五臂保持：P0、P-cap、P-history、W-correct、W-shuffled-original。相同原checkpoint、固定66参考、seed42/43/44、8192环境×24步×500PPO更新，各98,304,000真实转移，五臂共1,474,560,000。原PPO预算为论文2000轮的25%，不能称论文全预算；辅助优化计算和数据预训练成本单列。

|臂|检验问题|若该因素有效的预期|优先级|
|---|---|---|---|
|P0|额外真实PPO续训能解释收益吗|仅训练量有效时接近W|必须|
|P-cap|GRU部署容量/128D接口能解释吗|只容量有效时接近W|必须|
|P-history|历史记忆及PPO直接优化GRU能解释吗|预测监督无增益时接近W|必须|
|W-correct|正确动作条件多步残差表示是否有效|超过前三臂并过原物理控制门|主方法|
|W-shuffled-original|完整正确预训练包与原错误监督包差异|正确包有效时明显更好；不能独立归因在线内容|原预注册必须保留|
|W-shuffled-initmatched|从正确W同样模型/optimizer起点，仅线上donor错配|在线正确预测重要时比正确W差|原五臂主门通过后必须|
|W-pretrained-frozen|同正确预训练encoder，冻结且无线上aux|在线适应重要时弱于正确W，持平时仅能称预训练特征收益|原五臂主门通过后必须|

两个解释臂若需要运行，再增加589,824,000真实转移，总七臂2,064,384,000。没有实际吞吐诊断前不编造GPU小时。所有原五臂必须按相同完整预算结束，不能看中途好坏只停对照；除数值/系统/数据污染失败外，不按性能提前停单臂。正确W未过门就停止候选，不加容量/更新/推力强度。动作条件置零和H1-only仅在论文确实声称该因素带来控制收益时另作预注册实验，不用它们替换失败主臂。

## 时间、梯度与残差边界

每rollout冻结世界模型版本φ、输入归一化及Ridge。h只读历史本体o≤t与已执行动作a<t；不读当前待采动作、未来动作/状态、施力方向或时钟。缓存h/old-log-prob/真实reward/done/value及版本，末端value和GAE也用同版本；PPO全部epoch读取同一detached h，PPO梯度不进入世界模型。P-history/cap由PPO学习GRU，缓存原始有序输入并用当前参数算numerator、旧logprob算denominator，不缓存其latent。

PPO之后才作辅助更新，下周期重新从有序过去历史计算新h。Ridge及其归一化固定，不在线重拟合。当前新rollout在Ridge训练之外，因此线上无需OOF；连续残差目标为真实delta−当前行Ridge预测，contact独立BCE。W模型selected weights与末次optimizer不是同一时刻，两个W统一fresh Adam/schedule，不加载不匹配的末1000步optimizer；同起点解释臂复制正确selected weights和相同fresh Adam起点。

错误监督在task、clean/push、phase bin、contact层、推力方向及horizon内用可重建donor双射，donor不同reference/episode，历史/未来不重叠。错误连续目标为当前行Ridge＋donor残差，contact直接donor标签；不能打乱完整未来状态。donor不足行按同规则在correct/shuffled共同辅助采样排除，控制分母及reset/跌倒不删除。辅助窗必须完整10历史/H10未来且不跨reset；避免Ridge与encoder零填充mask差异。N_aux/batch/rng/horizon在只看运行吞吐的smoke后固定，不能看回报调整。

线上aux可能使下一rollout动作分布跳变，其变化不受上一PPOclip约束。交接前在固定训练审计history、actor不变时计算新旧encoder分布KL；上限为原desired_kl=0.01，非有限/超限判该run失败，不根据回报减小步数后续跑。

## 原算法恢复和公平性

原actor/critic完整Adam moments、step、LR/scheduler恢复；扩展首层原626列权重/moments照抄，新增128列为0，新增GRU参数新optimizer state。原626观测统计冻结；非特权critic使用actor-normalized观测，原unused privileged norm count0不喂critic。P-cap/history同GRU随机初始化、非仿射LayerNorm、相同warmup输入mean/std；P-cap当前本体重复10帧、前动作通道置零、每次hidden从零。分列部署/PPO/aux/总参数及计算，不用无效参数凑总量。

**Timeout保持官方近似**：经reviewer澄清同意，原RSL-RL为单次 `reward += gamma * transition.values * timeout`，value来自采动作时，auto-reset环境未提供terminal observation。五臂逐张量对齐原P0，不另算terminalvalue/重复bootstrap。它是沿用系统近似，不能称精确terminal-state bootstrap。真termination不bootstrap，所有termination/truncation/reset切断辅助窗。

旧11 test退休，不再作为控制validation或调参；固定66参考PPO不改。新11已参与方法级预测门，但不参加PPO/aux/norm，仅作固定update500后控制确认；不得称完全untouched端到端test。要提出更强独立泛化，须正式PPO前另锁controltest，不选择有利动作。

## 实施前核验与实际控制门

smoke必须核验初始mean/std/value/logprob等价、原优化器逐张量恢复/新列为0、actor不能看未来、WM冻结于PPOepoch、aux只更新encoder/head且Ridge/actor固定、epoch0ratio≈1、ordered history/reset、原timeout恰好一次、donor精确排列/边际、预算/墙钟/版本及交接KL。独立实际场景本地可视Gate仍必需，监控故障不得阻塞训练；旧checkpoint回放不能充Gate。

真实控制主门保持：正确W恢复AUC比P0/cap/history各改善≥10%且reference配对CI下界>0；优于原错配；成功损失≤5pp、跌倒不增、clean跟踪非劣5%，三seed方向一致。五臂PASS后才做两个解释臂，失败时收窄或否定主张。全部Isaac因果门通过后以统一update500冻结到MuJoCo，至少转移原五臂，不重调norm/Ridge/PD、不加IK，不只保留迁移成功臂。

这些要求尚未由正式runner执行。准备、开发预测或接口smoke都不是PPO控制创新验证完成。

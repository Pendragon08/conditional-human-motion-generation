# 世界模型＋PPO：一次有界残差开发对照

上一轮v1预测FAIL原样保留，commit e55daae；五臂正式PPO尚未启动。用户继续执行授权后，开展新的开发阶段，不能事后把旧test改成未见确认集。

独立审查由world_ppo_prediction_doc_witness_20261006按ablation-planner提供：加direct-GRU、未来动作置零两个支路、置换OOF残差、旧test退休、保留update0选择、梯度实际证据。这里落实这些要求；为最小成本，一次施力事件沿用已真实见证的torso80N/0.2s接口，而不扩为三事件或增加力矩/扰动强度。这只是扰动后的域内开发，不能涵盖随机扰动鲁棒性。OOF采用四个完整reference折。

数据：每seed42/43/44，原66参考、原冻结随机actor/std、8192环境480步。按原reference和起始phase0.2s排序、rank交替分干净／施力，总数各4096；每组奇偶差最多1，施力符号交替。step50..59世界y ±80N作用torso，10步16N·s计划冲量；显式外力不进入actor/encoder。奖励、PD、参考、termination不改；扰动分布新增，和v1 nominal分开计账。一次共享总真实采样11,796,480。实际施力buffer、方向、起始episode、done与reset记录；中断事件与跌倒不能从控制分母删除。

原44train/11validation保留，旧11test完全退休，本程序不评分它们。新data含这些refs但只使用前两组。每seed最大65,536train／16,384validation窗口，两域各半；phase/contact/施力方向/恢复0.2s组内精确跨reference/episode置换，无法匹配行所有模型共同剔除。恢复窗只采step60..109且属于push开始episode，H10未来无外力，历史允许包含已发生的force。跨reset窗口不合法，排除只为预测定义，不能称排除后存活率为鲁棒成功率。

六模型：Ridge-A、Ridge-0、原Direct-GRU-A、Residual-GRU-A、Residual-GRU-0、Residual-GRU-shuffled。未来实际动作仅作监督头条件；未来动作置零的独立Ridge-0和残差头都置零，历史上一动作保留。连续预测为冻结ridge＋GRU残差；contact仍独立BCE分类，避免logit残差。Ridge alpha0.1/1/10只validation选。用四折训练reference OOF预测求连续残差，最后部署预测ridge用全部train拟合；OOF不使用留折标签。归一化全部来自44train，OOF折内沿用全训练feature统计（不使用留折标签）。

错误监督精确置换OOF连续残差，contact二值标签随同donor。保持残差和contact标签边际，**不声称重构完整目标边际不变**。训练时使用当前行OOF baseline；正确对应本行残差，错配对应donor残差。评估时所有residual使用冻结全train ridge。相同初始化GRU、样本顺序，四GRU各1000Adam更新、batch1024，loss原48连续量统一std＋0.1contactBCE，不按小残差std放大；clip1、lr0.001。checkpoint从0/200/400/600/800/1000以validation恢复关节MSE选，所有模型仍执行1000更新。错误模型可选update0，不能人为迫使退化。每步实际CUDA梯度/finite/norm/非零参数数目、optimizer计数、初末参数hash及最终optimizer状态保存。

Ridge/OOF/固定linear支路另报计算和buffer参数；相同梯度数或真实步数不等于相同总计算。world encoder部署仅128维h，固定linear预测支路只训练/诊断使用。h不含未来动作、未来真状态、外力／方向／事件计时；这是PPO的候选辅助表征，不是模型想象rollout。

开发triage保留≥10%改善及reference配对CI下界>0（对Ridge-A及未来动作置零），clean FK不劣5%，恢复FK不比direct差5%，且优于错配残差。validation用于checkpoint选择，因此这些CI只作为开发诊断，不能当独立统计确认。失败就停止该残差候选，不加容量/更新/force强度追结果。

如开发三seed均改善：预先固定规则生成11条新的、未参与旧桥拟合/预测/结构选择的Kick确认参考，再单次确认。原nominal PPO协议要同时通过新clean确认的原预测门；仅recovery改善不足以启动nominal正式PPO，需另锁扰动域PPO假设。新确认集失败不重复test。无论开发结果怎样，此队列不会启动正式PPO。正式控制还须当前实际Kick本地画面通过，随后P0/P-cap/P-history/W-correct/W-shuffled三seed匹配更新，才可验证控制因果收益；最后冻结到MuJoCo。

root-frame H10 URDF FK与root线速度m/s、角速度rad/s、重力、根高度、足滑及地面contact AP/Brier分列。预测门不证明世界坐标目标或实体打击，Kick无实体目标。原world v1、额外IK否定结果、旧MPC失败全部保留。

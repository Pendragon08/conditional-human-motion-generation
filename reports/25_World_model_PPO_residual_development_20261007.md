# 世界模型＋PPO：残差动力学开发阶段（2026-10-07）

用户授权继续执行。上一轮三个种子的直接GRU预测均未优于强线性基线，原FAIL保留于报告24。新的三种子开发产物已完整生成，**目前没有正式PPO收益或创新点通过结论**。

## 本轮实际方法

扩散生成的66条官方Kick参考仍是运动跟踪目标；原始G1重定向不变，不加额外IK。原策略在真实Isaac Lab中执行，世界模型从可部署本体历史和过去动作学习物理响应。监督头可使用已经采集的未来动作序列，actor的当前latent不能使用未来动作或真实未来状态。

本次比较：强线性Ridge、未来动作置零Ridge、原直接GRU、线性＋GRU残差、未来动作置零残差、错误残差监督。正确／错误残差同架构和更新预算；错误监督精确置换训练reference四折OOF残差，保留连续残差及contact标签边际。contact采用直接二值分类，避免logit残差。update0可被选择，错误对照不会被强制训练到退化。

每seed42/43/44，8192真实环境×480控制步，各4096干净／施力环境；原随机actor、std、奖励、PD、参考及终止不改。新增外力±80N作用torso、持续0.2s，单独计为扰动开发分布。每seed只采一次，所有模型共享；总计划新增11,796,480转移，正式PPO更新为0。恢复预测窗只在外力结束后且未来0.2s没有新外力；reset仍记录，跨reset窗口排除只用于定义预测标签，不代表控制成功。

原44train与11validation继续开发，旧11test已曝光，完全退休。validation同时用于checkpoint选择，因此开发改善／CI不等于独立确认。若开发有收益，仍需新11条确认参考和单次确认门，然后才可以进入匹配PPO控制消融。具体锁定规则见plans/world_ppo_residual_development_20261007.md。

## 已核验与当前运行

- 16环境扰动接口由fresh文档见证原样执行一次：480步、7680真实转移，0 PPO／0aux，正常退出0。
- 10次Articulation力buffer：每次8个80N，buffer误差0；这是施力接口核验，不是控制收益。
- 真实reset4次，推力期0，恢复step60..109内1次；done与episode逐转移一致。11个数组大小／hash与有限性通过。
- 队列启动后真实CUDA残差数值见证通过：初始连续输出等于Ridge、future-action-zero两个支路均有效、有限非零辅助梯度。
- 唯一队列r3_world_residual_development_20261007按种子顺序collect→fit。每个GRU1000更新、batch1024；GPU0计算，GPU1独立实际状态渲染。监控不在训练依赖链中。

状态由artifacts/monitoring/r3_world_residual_development_20261007.state核对，产物在artifacts/world_residual_20261007。初始化没有真实step时不能报采集推进，队列启动不能报实验完成。

## 判定与产物

开发triage要求对Ridge及未来动作置零残差改善至少10%、reference配对CI下界>0，干净FK非劣5%，恢复FK不比direct差5%，优于错配。开发失败就停止残差候选，不能事后加容量／更新／力强度追好看结果。开发通过也不会自动启动正式PPO。

记录H1/H5/H10 root-frame URDF FK、关节误差、root线速度m/s／角速度rad/s、重力、根高度、足滑、contact Brier/AP及prevalence。地面接触不等于实体打击；根相对FK不验证世界坐标目标。正式控制还需真实恢复AUC、跌倒、任务成功及三seed匹配消融，最后冻结策略到MuJoCo。

服务器保存原数据、窗口、OOF fold coefficients、donor排列、世界模型、每步CUDA梯度审计、optimizer步数与状态、初末参数hash。GitHub归档代码、协议、结果metadata与报告，不上传许可原资产／权重。原DreamControl／CLoSD教师基线汇报保持独立。

## 独立文档与数据核查

fresh见证world_residual_development_doc_witness_20261007实际执行唯一tmux调用并只读审核：三个collect和三个fit包装器返回0、完成哨兵及产物完整、总11,796,480转移、5,121真实reset、12,000 CUDA辅助更新；156个文件的bytes/SHA均匹配。所有窗口/标签与原数组完全一致，跨reset/reference为0，全部donor精确分层排列，四折OOF系数重新计算prior误差0，normalizer来自44train且误差0，退休11参考未被评分。

补充只读审计发现开发包装器在timeout之后先赋值SENTINEL，再读取$?，因此实际子进程退出码未独立留存。六份日志无其他独立退出回执，采集完成哨兵位于环境清理之前，不能排除清理异常或后续超时。原执行源码已保全，不重跑、不改原state；计数、哈希、梯度与开发指标审计仍有效。新确认包装器在timeout返回后立即保存退出码，独立确认不沿用此错误。

施力后step60..109重置计数分别972/966/1002；大量重置仍须计入未来控制评价，预测有效窗口改善不代表恢复成功。错配模型三个种子均选择update0，但仍实际训练1000更新。state-only在seed43/44选择800；这些checkpoint不是末次训练权重，因此其末次参数hash只能由保存记录/optimizer及梯度证据支持，无法从末次权重文件独立重算。

旧非残差fitter的远程实际执行源码与本地后续修正存在三处报告差异：eligible窗口数记录、root线/角速度由旧混合量改成分列量。未改变原预测FK主要指标、模型拟合或旧FAIL结论。远程实际源码保存在results/world_residual_20261007/raw/previous_executed_r3_world_ppo_fit_v1.py；不把后续修正宣称已执行。新残差源码在见证时本地/服务器字节一致，线速度m/s和角速度rad/s分列。

<!-- AUTO_DEVELOPMENT_STATUS -->

## 自动核对状态（北京时间 2026-10-07 14:09）

开发对照已完成；stage=development_only。新采集去重转移11,796,480，完成拟合3/3种子。正式PPO与退休test调用均为0。

|seed|Ridge mm|Direct mm|Residual mm|Future-action-zero mm|Shuffled mm|对Ridge开发改善 [CI]|开发triage|
|---|---|---|---|---|---|---|---|
|42|13.285|12.697|10.469|28.174|13.285|22.01% [18.98%,25.03%]|PROMISING_NEEDS_NEW_CONFIRMATION_REFS|
|43|13.136|12.034|10.240|28.279|13.136|22.37% [19.76%,24.59%]|PROMISING_NEEDS_NEW_CONFIRMATION_REFS|
|44|12.818|12.024|9.933|28.629|12.818|22.70% [20.78%,24.51%]|PROMISING_NEEDS_NEW_CONFIRMATION_REFS|

validation选择checkpoint，本表不是独立确认。mm是全窗口聚合RMSE，改善/CI则逐reference的窗口MSE均值开方后等权平均，不能直接混算。

# 残差世界模型+PPO控制确认：运行前锁定的度量细节

此文在正式PPO及任何最终控制评分前锁定；承接plans/isaac_feedback_world_ppo_20261006.md、world_residual_ppo_ablation_review_20261007.md。预测确认PASS不代表控制PASS。五臂三seed固定原66参考各8192×24×500更新；首次从同一原Kick2000轮checkpoint和完整PPO Adam/LR继续，不是从零重训或论文2000轮严格复现。辅助预测数据、计算与参数成本单独列出。新11参考已用于预测门，未用于PPO、aux或norm，不称端到端未见test。

## 固定最终评分

只评各臂update500；无中途挑checkpoint、测试反馈调参、性能早停。2048环境×500帧×.02s、seed42/43/44、deterministic actor、原play设置corruption=False；所有norm/actor/critic/encoder/Ridge冻结，0PPO/0aux。五臂三seed共15,360,000评估转移，另列而不当训练步。结论只覆盖固定phase0条件，不覆盖随机phase和传感器噪声鲁棒性。

外生分配表按reference→clean/push→世界Y正负循环平衡，2048除11有余数，各层尽可能计数差<=1。同一seed的固定表供全部臂使用；初次reset按原重定向reference姿态与原0.15m起始高度设定，仅替换reference采样分配，奖励/PD/物理/后续原auto-reset保持。phase必须0。每reference含clean/push与两方向，不基于结果选取。push组step50..59对torso_link施世界Y ±80N，.2秒；clean组始终0N。力转body-frame并核验buffer，方向不进入policy或encoder。

Primary只统计push组step60..109。每步先算全部27跟踪关节的RMSE，再乘.02s求和，单位rad·s。在每reference内平均初始push环境AUC，再11reference等权macro。first-episode完整保留：step t的误差在动作执行前、尚未auto-reset时取；若该transition第一次done，该步仍用执行前实测，后续step全部用固定2rad失效插补。无法在done后读到精确terminal pose，不将reset新episode误差接入原轨迹。2rad是既有协议失效罚值，不是最大合法误差；若合法误差更大可能形成插补偏差，明确限制并附failure不增门、实测误差与存活时间。

Clean主非劣指标同60..109每步RMSE及同一first-episode失效插补。另报clean全500步RMSE/AUC、全身FK/root误差（描述性）、first-episode failure比例与存活时长。所有初始环境为分母，无删除摔倒、重置或未完成样本。

Kick原termination为超时、root高度或角度失败；不存在success termination。以原term manager名字逐步记录root/angle等failure，truncation/time_out另列；不把正常timeout当摔倒，多个失败条件同时命中只计一次first-episode failure。first-episode success包括首次done同一transition的n_successes增量；另外列官方any-reset `n_successes>0/N`公式结果。虚拟脚踝目标成功不等于实体打击/接触，统计参考事件触发而非文本语义。

## 固定比较与门

每seed独立reference配对bootstrap10000次，RNG20261007；每次重采样后重新计算macro比值。改善=(control−W)/control。W-correct恢复AUC对P0、P-cap、P-history各点估计改善>=10%且95%CI下界>0；对W-shuffled-original改善CI下界>0。Clean W/control对三个主要control都须95%CI上界<=1.05。任务保持相对P0，first-episode success下降<=5百分点与failure比例不增均是**点估计门**，同时报告reference配对CI但不临时替换门。三seed各通过，不通过时不池化seed救结果。

原五臂必须完整同预算；系统/数值/数据完整性失败保全并标明不能评分。主门失败则停止当前候选，不按结果加推力/容量/辅助次数。主门全部PASS才继续已预注册的同起点错配与冻结encoder两解释臂，仍各8192×24×500×3seed；五臂控制收益不能直接归因线上更新。所有Isaac因果/解释门通过后再统一冻结到MuJoCo，不调PD/norm、不加IK、不删除失败臂。

训练中每固定50更新的描述性日志含实际reward、value/surrogate/entropy、真实成功/reset、辅助loss/KL、真实步数、墙钟、显存。训练实际平均reward的曲线AUC为secondary；官方参考事件任意reset成功累积计数不等于held-out曲线，不能据此声称已达到预设独立测试成功率。原计划提及达到预设成功率的步数，但未锁具体阈值且没有独立中间测试，当前标为未测而不结果后补阈值。主评只用update500。独立审查建议记录的限制与所采纳规则均保留。

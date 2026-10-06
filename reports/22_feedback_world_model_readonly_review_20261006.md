# 首轮F/W消融的只读结果审计回执

日期：2026-10-06。由另行提示的同模型家族代理审阅源码、固定协议和最终结果；未修改文件、未新运行物理实验。本文件由主执行者记录其审计结论，**不是独立科学复现**。

- 新学习控制240个真实MuJoCo trial完整、唯一；配置、已有模型和oracle结果哈希匹配，队列正常exit0。
- 候选集合、H10、5步决策周期、干预区间、代价和真实评分与局部oracle匹配；同运行各臂完整初态克隆。历史初态由相同确定性prefix重建，没有跨运行逐参考字节哈希验真。批量PPO与逐样本PPO经验证数值等价，非逐bit相同。
- state-only忽略所有候选动作，预测跟踪代价一致；正残差惩罚使零候选最优，解析复用PPO合理。这不是新执行的经验臂。
- **正式learned_control_gate=FAIL正确。**主臂对PPO点改善15.21%，但95%CI[-4.72%,28.33%]；对simple1.07%，CI[-16.12%,13.87%]；跌倒42.5%高于PPO40%。不允许换用一步监督辅助臂来改变主判定。
- Kick主臂恢复AUC较PPO退化；Punch出现积极点信号，总体不能推广到两任务。五步与一步的控制差异CI跨零，预测多步收益不等于控制多步收益。
- 主臂比同候选oracle的恢复AUC高20.54%；这表明本实现未兑现已见局部前瞻空间，不能直接定位为某个预测、外推或容量原因。
- 候选评分平均72.32ms、最大250ms，25/2132超过100ms。只计候选评分核，不含主PPO/物理/通信/I/O；非无漏期在线、50Hz或实机实时证明。
- 训练seed与候选seed绑定；reference bootstrap平均种子，不能充分覆盖模型和采样方差。所有阶段复用相同40参考，是分阶段pilot，非独立最终确认。
- 合理停止：保留replay预测PASS、局部oraclePASS、学习控制FAIL三层结论；停止当前接口扩张，不重训或调测试门槛，不启动F×W或正式PPO。F仍只有慢速稀疏生成响应，缺少完整前缀和后续真实机器人控制。

源结果：[正式240回合](../results/feedback_world_model_20261006/learned_control_summary.json)、[主执行者复核及事后诊断](../results/feedback_world_model_20261006/learned_control_reviewer_diagnostic.json)、[最终阶段回执](../results/feedback_world_model_20261006/final_status.json)。诊断文件`status=PASS`仅指完整性，不覆盖控制门FAIL。

# 跟踪训练计数与参考来源更正

日期：2026-10-17。此说明补充已冻结的计划、执行源码、收据和上一轮报告，不改写历史证据。

跟踪实验 `optimizer_updates`、`parent_optimizer_updates`、`additional_updates_requested`、汇总 `updates` 和检查点附件名中的 updates，均表示 **PPO learning iterations（采样—学习轮次）**，并非一次 `optimizer.step()`。同一轮先采样 1024×24 个环境转移，再进行 5 个 epoch×4 个 minibatch，即 20 次 Adam 优化器步进。

|阶段|PPO 轮次|对应 Adam 步进|环境转移|
|---|---:|---:|---:|
|父模型训练|500|10,000|12,288,000|
|本轮续训|1,500|30,000|36,864,000|
|最终累计|2,000|40,000|49,152,000|

以上 Adam 计数由实际保存配置推导，并以 `results/continuation_2000/state_comparison.json` 中检查点优化器状态核对。此处统计的是优化器步进轮数，不将不同参数的计数相加。独立 1 轮恢复冒烟实验不计入正式续训预算。生成入口的两次人工张量测试是两次实际 AdamW 步进，与跟踪轮次命名无关。

评估脚本历史元数据中的 “not model output” 应精确解读为 **not output of the evaluated policy**。参考轨迹来自外部视频重建、GMR 重定向和 G1 前向运动学，上游处理可能使用学习模型；它是独立的运动学代理参考，不是实测机器人真值。该结果分类为 `simulation_only`。

新报告和图表使用 PPO 训练轮次；历史附件名保留以维持下载地址与哈希的可追溯性。

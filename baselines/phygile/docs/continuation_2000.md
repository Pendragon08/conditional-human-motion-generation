# 2000 轮 PPO 训练续训协议

日期：2026-10-17。沿用 `docs/environment.md` 中 spec `fa5883c8` 的现有服务器与依赖，不修改共享安装。服务器目录仍为 `/data/code/ouyanghao/oyh/phygile-reproduction-20261017`。

目的：在同一组 20 个训练动作、同一奖励和网络下，从已归档的 500 轮 PPO 训练检查点再训练 1500 轮 PPO，检验增加预算的效果。固定最终检查点评估；不根据测试分数选择训练时长或检查点。测试集已经在前一轮查看，因此本次仅为探索性续训比较。

恢复时逐张量检查策略、优化器与观测归一化状态；将下一轮索引设置为已完成索引加 1，并使自适应学习率标量与恢复的优化器一致。仿真、随机数及采样器状态重新初始化，不能称为逐比特连续训练。此前 500 轮 PPO 训练结果与代码不改写。

独立文档运行验证，必须原样执行以下命令（32 个环境、只增加 1 轮 PPO 训练；预期检查点索引 500）：

```sh
cd /data/code/ouyanghao/oyh/phygile-reproduction-20261017
bash scripts/train_extended_2000.sh 1 32 resume_validation2000
```

预期：退出码 0，日志含 `PHYGILE_VERIFIED_RESUME`，策略/优化器/归一化与父检查点一致；只激活 1 个专家；在 `logs/rsl_rl/phygile_moe_pilot/*resume_validation2000/` 产生 `model_500.pt` 和 `resume_provenance.json`。这项验证不证明策略已收敛。

正式续训（不应在独立验证时运行）：

```sh
python3 scripts/run_recorded.py --name continuation_2000/train --timeout 10800 -- bash scripts/train_extended_2000.sh
```

训练完成后使用 `scripts/evaluate_continuation.py` 在预先固定的 8 个片段上评估最终检查点，每片段 32 个首回合。继续使用原评价脚本和原始状态重算，不放宽终止条件。正式论文结果仍需要匹配的任务、多个来源组和多个训练种子，以及缺失的完整生成模块。

## 从 GitHub 附件在新目录复跑

先按主 README 创建干净工作目录、配置仿真依赖与机器人资产，并解压原始输入附件。再从 [500 轮 PPO 训练实验附件](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/phygile-tracking-pilot-20261017) 下载 `phygile_initial_expert_500updates_seed42.pt`。在新的工作目录执行：

```sh
python3 scripts/import_parent_checkpoint.py /path/to/phygile_initial_expert_500updates_seed42.pt
python3 scripts/run_continuation_pipeline.py
```

导入程序只接受固定 SHA-256 的父检查点，放到续训脚本所需的相对日志路径；不会覆盖不同的检查点。流水线依次执行 1500 轮额外 PPO 训练、8 组固定评估和原始数据重算。`results/continuation_2000/pipeline.json` 的 `COMPLETED` 与所有阶段退出码为 0 才表示该次流水线完成。

这条路径复现的是从已归档权重继续训练的过程。若从头训练 500 次，受仿真非确定性影响，新检查点通常不会有相同哈希，不能直接冒充本次固定父检查点。应为新的从头训练实验另建计划与来源记录。

## 最终状态核对与计数更正

上述三阶段流水线完成后，还必须执行独立的状态检查并保留收据；生成报告要求该收据成功和 `state_comparison.json` 为 PASSED：

```sh
python3 scripts/run_recorded.py --name continuation_2000/state_comparison_check --timeout 120 -- bash scripts/run_isaac.sh scripts/verify_continuation_states.py
```

历史字段 `optimizer_updates` 和附件名 updates 表示 PPO 轮次，而非 Adam 步数。每轮 5 epoch×4 minibatch=20 次 Adam 步进，详见 [更正说明](accounting_terms_erratum.md)。冻结计划、执行源码和原始收据保留原样。

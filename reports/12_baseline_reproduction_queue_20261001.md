# 论文基线恢复执行记录（2026-10-01）

本轮按用户指示继续论文复现，暂停创新点扩展。新结果仅写入 conditional-human-motion-generation 项目及其私有 GitHub。

DreamControl 原论文确实包含 Precise Kick 和 Precise Punch，官方任务名分别为 Kick 和 Punch。已完成官方预算的控制器基线：Kick85.3%、Punch87.2%。其余九项的输入已核验，唯一顺序训练/评估队列已启动，首项Jump推进到第324/2000轮（14:29 UTC）。每项8192环境、2000迭代、seed42，随后官方1000环境500帧评估和视频。尚未完成的任务不记为失败率0%。

CLoSD旧512环境运行保存epoch66000后异常退出139；并非没有训练到目标。原日志没有原生栈，不能确定具体清理调用。学习率2e-5与官方一致；512环境每轮样本量只有官方3072环境的1/6是已知差异，但未证明是唯一退化原因。新的干净官方源码、独立监控和退出诊断队列已启动，先检查3072环境三次更新及正常退出，再做4096环境官方权重评估及正式3072环境微调。当前初始化阶段尚未出现epoch日志，不宣称正式4000轮微调已启动或复现完成。

两项运行互不依赖本地画面连接。DreamControl actual-state页面6013已核验新帧；CLoSD7001已恢复本地隧道，但只有接收到新运行实际数据并验证当前场景后才通过正式训练可视Gate。旧预览不是当前训练。

可复查记录：

- `scripts/r3_dreamcontrol_remaining_queue.py`、`docs/remaining_baselines_execution.md`、`results/r3_dreamcontrol_remaining_manifest.json`。
- `scripts/r3_closd_isolated_entry.py`、`scripts/r3_closd_reproduction_20261001.sh`、`scripts/r3_closd_scene_receiver.py`、`docs/closd_reproduction_20261001.md`。
- 正式成绩逐项汇入报告01/03和results；检查点/资源保留服务器，不上传许可受限文件。

源文献：[DreamControl原文](https://genrobo.github.io/DreamControl/static/paper/DreamControl_arxiv.pdf)、[CLoSD官方复现说明](https://github.com/GuyTevet/CLoSD#train-your-own-closd)。

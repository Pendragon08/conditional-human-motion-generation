# DreamControl 多任务基线复现记录

本页区分论文公布的数据、已经完成的本地实验和待运行任务。上游依据为 [DreamControl 论文](https://genrobo.github.io/DreamControl/static/paper/DreamControl_arxiv.pdf)及固定提交 `ec28b508313d6837de52c68247fbc8cf81fe4935`。论文的 Pick、Bimanual Pick、Ground Pick、Button Press、Open Drawer、Open Door、Punch、Kick、Jump、Sit 共十一项任务由独立策略复现，不能把 Kick 成绩推广为通用生成能力。

## 已完成

- Kick：官方 66 条动作、8192 个并行环境、2000 次迭代、seed 42；官方 1000 环境、500 帧评估成功率 85.3%。模型、指标和视频已归档，见 `reports/02_dreamcontrol_reproduction.md`。
- 其余十任务的公开参考动作已从上游 Git LFS 取得并逐文件记录大小与 SHA-256，见 `results/official_other_tasks_assets_manifest.json`。此处仅证明参考输入完整，尚不代表训练、场景资产或评估通过。

## 待运行的同口径复现

`scripts/r3_dreamcontrol_official_other_tasks_queue.sh` 顺序处理 Punch、Jump、Sit、Pick、Bimanual Pick、Ground Pick Side/Top、Button Press、Open Drawer、Open Door。每项使用官方任务配置与参考动作、8192 环境、2000 次迭代、seed 42；训练完成后按官方 `play_eval.py` 的 1000 环境、500 帧口径保存成功率和非空视频。单项失败即停队列并保留日志，不把未完成任务记为 0% 成功率。

2026-09-27 启动了 Punch 的 **64 环境图形预检**，用于检查实际训练仿真是否能够串流，且不改动 GPU 1 正在运行的 CLoSD。预检推进至第 385 轮后按 30 分钟时限停止，遗留的 Kit 子进程已单独清理；该预检不是论文参数基线，其训练日志不能作为 Punch 正式成绩。用户随后明确回复“可以开始”，但没有明确报告当前 Punch 画面在本地已显示，因此 Gate 中分别记录“授权启动”和“本地画面未明确确认”，不伪造可视确认。

2026-09-27 07:19 UTC 从独立服务器目录 `/data/code/ouyanghao/oyh/conditional-human-motion-generation` 启动正式十任务队列。首项 Punch 使用 GPU 0、官方 100 条参考动作、8192 环境、2000 轮与 seed 42。Isaac Sim 在新图形缓存中首次编译着色器并建立 8192 个场景后，**07:31 UTC 出现第 0/2000 轮训练日志**，当前尚无最终成功率。GPU 1 的 CLoSD 仍在原运行目录继续。正式队列的画面与训练耦合仍有 WebRTC 停滞风险，应以迭代更新时间和检查点持续监测，不能仅凭网页/端口可连判断训练正常。用户是否在本地看见本次 Punch 正式训练画面仍待反馈。

复现的下游任务成功率用于比较控制执行结果；本课题的生成模型仍须单独评价文本/空间约束满足度、动作多样性、物理质量及跨条件泛化，不能由这些策略成功率单独证明。

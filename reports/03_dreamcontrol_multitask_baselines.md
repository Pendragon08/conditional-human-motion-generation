# DreamControl 多任务基线复现记录

本页区分论文公布的数据、已经完成的本地实验和待运行任务。上游依据为 [DreamControl 论文](https://genrobo.github.io/DreamControl/static/paper/DreamControl_arxiv.pdf)及固定提交 `ec28b508313d6837de52c68247fbc8cf81fe4935`。论文的 Pick、Bimanual Pick、Ground Pick、Button Press、Open Drawer、Open Door、Punch、Kick、Jump、Sit 共十一项任务由独立策略复现，不能把 Kick 成绩推广为通用生成能力。

## 已完成

- Kick：官方 66 条动作、8192 个并行环境、2000 次迭代、seed 42；官方 1000 环境、500 帧评估成功率 85.3%。模型、指标和视频已归档，见 `reports/02_dreamcontrol_reproduction.md`。
- Punch：官方 100 条参考动作、8192 环境、seed 42，从第 1600 轮检查点完成余下 400 轮，最终第 1999/2000 轮检查点已核验；官方 1000 环境、500 帧评估成功率 **87.2%**，非空视频与 SHA-256 已核验。结构化结果见 `results/r3_dreamcontrol_official_punch_8192_eval.json`，视频见 `media/r3_official_punch_8192_eval.mp4`。这是控制器基线，不是本课题新生成模型的成绩。
- 其余十任务的公开参考动作已从上游 Git LFS 取得并逐文件记录大小与 SHA-256，见 `results/official_other_tasks_assets_manifest.json`。此处仅证明参考输入完整，尚不代表训练、场景资产或评估通过。

## 待运行的同口径复现

`scripts/r3_dreamcontrol_official_other_tasks_queue.sh` 顺序处理 Punch、Jump、Sit、Pick、Bimanual Pick、Ground Pick Side/Top、Button Press、Open Drawer、Open Door。每项使用官方任务配置与参考动作、8192 环境、2000 次迭代、seed 42；训练完成后按官方 `play_eval.py` 的 1000 环境、500 帧口径保存成功率和非空视频。单项失败即停队列并保留日志，不把未完成任务记为 0% 成功率。

2026-09-27 启动了 Punch 的 **64 环境图形预检**，用于检查实际训练仿真是否能够串流，且不改动 GPU 1 正在运行的 CLoSD。预检推进至第 385 轮后按 30 分钟时限停止，遗留的 Kit 子进程已单独清理；该预检不是论文参数基线，其训练日志不能作为 Punch 正式成绩。用户随后明确回复“可以开始”，但没有明确报告当前 Punch 画面在本地已显示，因此 Gate 中分别记录“授权启动”和“本地画面未明确确认”，不伪造可视确认。

2026-09-27 07:19 UTC 从独立服务器目录 `/data/code/ouyanghao/oyh/conditional-human-motion-generation` 启动正式十任务队列。首项 Punch 使用 GPU 0、官方 100 条参考动作、8192 环境、2000 轮与 seed 42。Isaac Sim 在新图形缓存中首次编译着色器并建立 8192 个场景后，**07:31 UTC 出现第 0/2000 轮训练日志**，当前尚无最终成功率。GPU 1 的 CLoSD 仍在原运行目录继续。正式队列的画面与训练耦合仍有 WebRTC 停滞风险，应以迭代更新时间和检查点持续监测，不能仅凭网页/端口可连判断训练正常。用户是否在本地看见本次 Punch 正式训练画面仍待反馈。

复现的下游任务成功率用于比较控制执行结果；本课题的生成模型仍须单独评价文本/空间约束满足度、动作多样性、物理质量及跨条件泛化，不能由这些策略成功率单独证明。

2026-09-29 状态更新：上文 2026-09-27 “尚无最终成功率”描述的是当时进度。首轮 Punch 在约第 1618 轮因 WebRTC/GLFW 停滞，校验并保全 `model_1600.pt` 后，将训练与独立检查点预览分离，从第 1600 轮续训余下 400 轮并完成正式评估。原十任务顺序队列在 Punch 处中断；除 Punch 与此前 Kick 外，其余九项仍未训练，不能写成已完成或失败成绩。

## 2026-10-01：其余九项正式队列已启动

用户要求先继续论文基线复现。已排除完成的 Kick/Punch，核验九项原始参考文件数量与 SHA-256，并启动唯一队列 `r3_dreamcontrol_remaining_20261001`。顺序为 Jump → Sit → Pick → Bimanual-Pick → Ground-Pick → Ground-Pick-Top → Button-Press → Open-Drawer → Open-Door。各项 GPU1、8192 环境、2000 PPO 迭代、seed42，从零训练，成功完成后自动执行官方 1000 环境、500 帧评估和视频核验；失败即停止，不从零重复训练、不悄悄缩小环境数。

首项 Jump 已进入实际 PPO，2026-10-01 14:29 UTC 核对到第 324/2000 轮，尚无最终评估成绩。其余八项在同一队列等待，排队不计完成。状态和日志前缀为 `artifacts/monitoring/r3_dreamcontrol_remaining_20261001`，输入清单为 `results/r3_dreamcontrol_remaining_manifest.json`。启动器 `scripts/r3_dreamcontrol_remaining_queue.py` 已通过独立按文档核验。

训练使用 headless 的官方 Isaac Lab 任务，保留原网络、奖励、PPO 和参考数据；仅复用已经验证的只读、非阻塞遥测入口。独立 Isaac Sim 渲染进程随当前任务切换场景，本地 http://127.0.0.1:6013/ 显示当前实际训练状态抽样与轮次。2026-10-01 本地状态接口核验到 Jump 第 280 轮，帧龄约 1.8 秒、状态步数持续增加。不是检查点回放，不是全部环境逐帧直播。画面或本地连接失败不阻塞训练。保留此前用户可视确认，不伪造本轮新的用户确认。

截至该九任务队列启动时正式完成 **2/11**（Kick 85.3%、Punch 87.2%）；九项最终结果需要逐项检查第 1999 轮模型、正常退出、官方成功率、非空视频和哈希后再计入。九项队列的计划不代表所有项目已经达到论文数值，后续完成情况以本报告最后的日期记录为准。

2026-10-01 15:42 UTC自动核对：Jump第1180/2000轮，日志最后更新15:41:48 UTC，GPU1持续计算，训练与独立渲染均保留。没有最终成功率，后续八项仍在同一队列等待；没有重复启动或修改训练。

## 2026-10-01 17:44 UTC：Jump完成，Sit已自动接续

Jump已正常完成第1999/2000轮，并以官方1000环境、500帧评估得到 **100.0%**。训练8192环境、2000迭代、seed42，从零训练；run为 `2026-10-01_14-01-43-Jump-a`。最终模型SHA-256为 `21b2525f2eab462a3514653b7e6c389b10ebf768c1aa154dd487640011e143aa`，独立读取与队列结果一致。原始参考清单中的Jump只有1条参考动作；成绩是官方Jump任务判据下的控制成功率，不证明对未见动作或通用文本/空间条件的生成质量。

结果在 `results/r3_dreamcontrol_official_Jump_8192_eval.json`。保存的训练配置在 `results/configs/DreamControl_Jump/agent.yaml` 与 `env.yaml`。正式评估视频已下载到 `media/r3_official_jump_8192_eval.mp4`，SHA-256为 `323c3283f5e82b2e05c5a3d59472bc57cf0349e4aa2f74502ccb3b3be914670b`，本地校验与服务器一致；1280×720、50fps、501帧、10.02秒，中段帧经视觉检查可见实际机器人与场景，非空白文件。评估日志最后记录 `Success rate: 100.0 %`，本地及服务器均保存该日志。

因此目前正式完成 **3/11**：Kick85.3%、Punch87.2%、Jump100.0%。同一唯一队列已自动启动Sit，17:44 UTC核对第563/2000轮，日志连续更新。Sit后仍有七项等待；不重复训练已完成任务，不修改或中断当前运行。CLoSD已完成的两组官方权重评估和待确认画面的新正式微调状态保持不变。

## 2026-10-01 19:46 UTC：Sit训练完成，首次评估异常

Sit已正常训练到第1999/2000轮，最终模型位于 `2026-10-01_16-54-27-Sit-a/model_1999.pt`，SHA-256 `4302e29fe6376452a21fe93351e616fb23ec5bb83206eb015df3b5be4bea85c7`。训练8192环境、2000轮、seed42，未改变预算。首次官方1000环境500帧视频评估在仿真初始化时触发 `malloc(): invalid size (unsorted)` 并退出1，尚未加载策略、未报告成功率；不记为0%，也不计入已完成训练加评估的任务数。

当时宿主内存241GiB可用，GPU显存仍有余量；不是已经证实的CUDA OOM，具体底层原因待查。队列已按失败保护停止，后续七项没有因此开始。失败评估日志SHA-256 `4879e262699ef61616078e0fe67423d32ac051ea3632ee33a1793cb9ad33e5ea`；训练日志SHA-256 `e7f1847490d3928cbc463612a7174df3a0c22f3285d47f8e98fde1dbb03ca4c1`。

恢复流程见 `docs/sit_eval_recovery_20261001.md`：只重试一次同模型、同官方参数评估，用独立文件保留新尝试，备份最终模型和原始日志/阻塞状态，不重训Sit；正常评估及视频验证通过后才允许原唯一调度器跳过完成任务并继续Pick。重复失败则保留阻塞，不自动循环。官方评估的环境seed未设置，应与训练seed42区分。
19:50 UTC一次干净的同参数、同模型官方评估重试仍在相同位置触发相同native malloc错误，退出1。重试状态 `artifacts/monitoring/r3_dreamcontrol_sit_eval_recovery_20261001.state` 为BLOCKED。原模型与日志已复制到服务器 `artifacts/recovery/Sit_20261001_final1999` 并校验哈希。独立文档验证仅检查输入和锁，exit0，不代表仿真成功。原训练和七项后续任务均没有重启。

随后仅做一次120秒上限的无摄像头初始化诊断，仍使用官方Sit入口、1000环境、同最终模型，区别只有未开启视频/相机，记录 `artifacts/monitoring/r3_sit_no_camera_init_diagnostic_20261001.*`。这是定位初始化故障的诊断，绝不计入500帧正式成绩；预期时限停止也不称正式训练失败。
## 2026-10-01 20:00 UTC：Sit评估恢复，Pick已接续

无摄像头1000环境诊断正常加载模型，按120秒期限停止（exit124、无native malloc错误）；不计正式成绩。之后使用官方 `play_eval.py --disable_fabric` 以USD I/O代替Fabric场景读写，保留同一模型、物理设置、原始参考、1000环境与500帧的视频评估，正常退出0，得到 **100.0%**。未重训Sit、未修改PPO或奖励，但显示读写参数与默认不同，明确记录为运行适配，而非完全相同运行配置的数值复现。具体native heap错误原因仍未证实。

最终模型与备份哈希一致；视频 `media/r3_official_sit_8192_eval.mp4` 为1280×720、50fps、501帧、10.02秒，729611字节，SHA-256 `4bc23e77d53e2e793ba70c20d2f74a7dec68c4c3cb07f287a2d7026a9cb1afe1`；本地和服务器校验一致。正式日志SHA-256 `22c65a7650f4ffa8a49ee72fd2e3412bcc83e63539ba517f07af04211077d714`。原两次失败记录及120秒诊断保留。训练配置、结构化结果分别在 `results/configs/DreamControl_Sit/`、`results/r3_dreamcontrol_official_Sit_8192_eval.json`；恢复代码、文档均通过独立输入检查，不能把该检查代替真实评估。

官方Sit使用1条参考，`touch_goal`对骨盆距离参考目标小于0.05米、且参考交互状态发生转换时累加成功；最终指标统计每个环境是否至少发生过一次成功。它没有单独检验持续坐垫接触或自然坐姿。中段视频可见部分机器人在座椅上方或上面，因而100%只能按上述官方判据解释，不能宣称所有坐姿语义均正确。

当前已完成 **4/11** 项官方控制任务（Sit含明确USD I/O适配），不是整篇论文复现完成；原唯一队列在评估正常退出后跳过已完成Jump/Sit并自动进入Pick。20:00 UTC核对Pick第3/2000轮、实际状态监控LIVE，后续六项等待。CLoSD正式3072微调仍等待此前已询问的当前实际场景可视确认，不重复询问或重新启动预检。
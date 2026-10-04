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
## 2026-10-02 00:46 UTC（北京时间08:46）：Pick完成，双手拾取接续

Pick官方67条参考、8192环境、2000轮、seed42从零训练，正常完成第1999/2000轮；官方 `play_eval.py` 默认Fabric路径、1000环境、500帧视频评估正常退出0，结果 **93.9%**（原始浮点93.90000700950623）。不沿用Sit的USD显示适配，未更改物理、奖励、网络、PPO或参考数据。

Run为 `2026-10-01_19-59-36-Pick-a`，最终模型SHA-256 `d3f34ca02946e7a56fec95f113e6646724c0dc8fc219437b95bbafcd31e8dcb4`，独立读取与队列一致。结果 `results/r3_dreamcontrol_official_Pick_8192_eval.json`；原始训练配置 `results/configs/DreamControl_Pick/agent.yaml`、`env.yaml`。模型保存在服务器，未重复训练或下载大型模型。

视频本地 `media/r3_official_pick_8192_eval.mp4` 为1280×720、50fps、501帧、10.02秒，1420105字节，SHA-256 `0941418de47a28b9bd9814189c5cd41abe7cb12f9cd07ce6694237145fae4b2a`，服务器与本地一致。中段帧可见机器人、桌面和物体，非空白。评估日志SHA-256 `b786c0a8725efb905e9274a59138ef10825129455746f5cf5eb3bb609a0b8f68` 已归档。成绩为官方任务判据，不能扩展为未知文本/空间条件生成、实机抓取或MuJoCo迁移通过。

已完成 **5/11** 项控制任务：Kick85.3%、Punch87.2%、Jump100%、Sit100%（显式USD显示适配）、Pick93.9%。同一唯一队列自动启动Bimanual-Pick，00:46UTC核对第207/2000轮，日志和实际监控连续更新；随后Ground-Pick、Ground-Pick-Top、Button-Press、Open-Drawer、Open-Door五项等待。CLoSD的新3072正式微调仍待此前已询问的可视确认，未启动或重复询问。
## 2026-10-02 04:49 UTC（北京时间12:49）：双手拾取训练完成，默认视频评估异常

Bimanual-Pick已正常到第1999/2000轮，官方1条参考、8192环境、2000轮、seed42，run `2026-10-02_00-21-24-Bimanual-Pick-a`。最终模型SHA-256 `70d025af972fb3b1a3bb9534dbb938f838484b4233111045b53cc09061c9c1ab`，训练日志SHA-256 `d42f1d44ddc1b4098424fb649ed5438f23a9756975aefe17f4b1869e4e25cba3`。默认视频评估在仿真初始化时native `malloc(): invalid size (unsorted)`，退出1，尚无成功率；不能记为0%或正式任务完成。失败日志SHA-256 `14aabfb7c0578697c11cf8d7555f4a9a8bf7b71ab3de70053e2f82a2b875e01b` 已留档。

宿主可用内存238GiB、GPU显存有余量，底层错误具体根因尚未证实。队列按失败保护停止，后五项未启动。依据Sit的既有可复查恢复经验，新的通用评估恢复脚本只做一次官方 `--disable_fabric` USD显示读写适配，保持模型、物理/奖励/参考和1000环境500帧预算，不重训或缩容；最终模型和原日志/阻塞状态备份并验hash。独立文档前置检查exit0，不能替代真正的仿真评估。文档 `docs/bimanual_eval_recovery_20261002.md`，状态 `artifacts/monitoring/r3_Bimanual-Pick_usd_eval_recovery_20261002.state`。
## 2026-10-02 04:55 UTC（北京时间12:55）：双手拾取评估恢复，地面拾取接续

同一Bimanual-Pick最终模型使用官方 `play_eval.py --disable_fabric` 的USD场景读写，1000环境、500帧，正常退出0，官方成功率 **99.9%**（原始浮点99.9000072479248）。未重训、未改PPO/奖励/参考/物理预算，但必须注明相对默认Fabric路径存在运行适配，不能称完全相同运行配置。第一轮native初始化错误具体原因仍未证实。

最终模型与备份的SHA一致；视频 `media/r3_official_bimanual_pick_8192_eval.mp4` 为1280×720、50fps、501帧、10.02秒、1171662字节，SHA-256 `87dbb8ec200dba90726fc5edee91287c0fbf09079fddf1178de1095b8201d088`，本地/服务器一致。代表帧可见机器人、桌面与物体，非空白。正式日志SHA-256 `9aa691e78d102da5832ee5fb9f52d4176369f4aea57a5897c73ceb506a9d8ee6`。结果及原始训练配置在 `results/r3_dreamcontrol_official_Bimanual-Pick_8192_eval.json`、`results/configs/DreamControl_Bimanual-Pick/`；失败记录单独留档，模型与原始日志备份在服务器。

成绩按官方任务判据统计，只使用1条原参考。代表画面有机器人位于桌面附近/上方，不能仅凭99.9%断言真实双手抓握稳定、物体搬运或全部姿态语义质量通过。本次没有新增MuJoCo sim-to-sim、未知条件生成或创新点验证。

当前正式完成 **6/11** 项控制任务（Sit和Bimanual-Pick明确记录USD I/O适配）。原唯一队列已自动进入Ground-Pick，核对第3/2000轮；后续Ground-Pick-Top、Button-Press、Open-Drawer、Open-Door四项等待。新恢复启动器 `scripts/r3_dreamcontrol_completed_eval_recovery.py` 完成评估后调用原未修改调度器，stdout为 `artifacts/monitoring/r3_Bimanual-Pick_usd_eval_recovery_20261002_queue.log`，锁和task状态前缀仍 `r3_dreamcontrol_remaining_20261001`。CLoSD正式3072微调等待此前的本地可视确认，不重复询问。
2026-10-02：已补完 Jump、Sit、Pick、Bimanual-Pick 的冻结策略 MuJoCo sim-to-sim，共70条参考、5段视频。Jump摔倒；Sit达参考目标但无持续座椅接触；Pick与Bimanual均未完成相对抬升5厘米。测试执行完成不等于任务迁移通过，单参考任务也没有泛化成功率结论。原Isaac自动重置成绩与本次单参考无重置口径不能直接比较。真实接触检测的正负夹具检查、原策略张量/观测接口核验均通过，但不能替代任务指标；Sit沿用训练座椅位置，与参考目标并未对齐。见[跨仿真报告](14_Completed_tasks_MuJoCo_sim2sim_20261002.md)。

## 2026-10-02 09:49 UTC（北京时间17:49）：Ground-Pick训练完成，评估恢复中

Ground-Pick的官方1条参考、8192环境、2000轮、seed42训练已正常完成1999/2000，run `2026-10-02_04-54-43-Ground-Pick-a`。最终模型SHA256 `0b853c6700957045cda3cb768f7d4ffc2ca7cfa67e79fc5fe33d8574f602e699`。默认1000环境500帧视频评估于08:33 UTC在仿真初始化报 `malloc(): invalid size (unsorted)`，包装器退出1，未产生成功率；具体native错误根因未证实，不能记0%或正式控制任务完成。训练、原失败日志与阻塞状态已备份至服务器 `artifacts/recovery/Ground-Pick_20261002_final1999` 并核验哈希。

新的任务专用文档为 `docs/ground_pick_eval_recovery_20261002.md`；独立新进程只读检查已exit0，确认模型哈希、旧队列不存在、单锁可获取、实际可视Gate已有确认与适配日志尚不存在。在原唯一tmux会话中仅启动一次既有通用恢复脚本，采用官方 `--disable_fabric` USD场景I/O，保持模型、物理/奖励/参考与1000环境500帧预算不变。此时只是恢复评估正在初始化，尚无最终分数；成功后自动恢复原调度器进入Ground-Pick-Top。若再次失败则停止，不重复重训或无限重试。原失败记录在 `results/r3_dreamcontrol_Ground-Pick_eval_failure_20261002.json`。正式完成仍为6/11，CLoSD新3072训练仍等待之前已询问的本地当前实际画面确认。

## 2026-10-02 09:51 UTC（北京时间17:51）：Ground-Pick完成，Ground-Pick-Top接续

同一Ground-Pick最终模型经上述唯一官方USD I/O适配，1000环境、500帧评估正常退出0，成功率 **38.1%**（原始浮点38.10000121593475）。这是本次保留的控制基线结果，不意味着达到论文效果或可靠地面抓取。原1条参考没有未见动作泛化结论，也没有新增MuJoCo验证。训练8192环境2000轮seed42不变，未缩容/重训/修改物理、PPO、奖励和参考；明确是运行适配，非完全同运行配置。

视频 `media/r3_official_ground_pick_8192_eval.mp4` 为1280×720、50fps、501帧、10.02秒，SHA256 `b310a2806d730d80fc5f33c6d0443a856514b85da6feea2c2b48abc9470cdfba`。正式评估日志SHA256 `96be836556d517e4278caf650c0932666312943be317155a07569d984aed6eb6`；原失败日志和备份仍保留。结构化指标见 `results/r3_dreamcontrol_official_Ground-Pick_8192_eval.json`，训练配置另存 `results/configs/DreamControl_Ground-Pick/`。最终模型SHA256仍为 `0b853c6700957045cda3cb768f7d4ffc2ca7cfa67e79fc5fe33d8574f602e699`。

当前已完成 **7/11** 控制任务：Kick85.3%、Punch87.2%、Jump100%、Sit100%、Pick93.9%、Bimanual-Pick99.9%、Ground-Pick38.1%。Sit、Bimanual-Pick、Ground-Pick均注明默认视频路径失败后使用官方USD I/O适配，不将native初始化失败记成0%。原唯一队列自动接续Ground-Pick-Top（09:51核对已加载Actor/Critic，尚未据此声称PPO迭代推进），其后Button-Press、Open-Drawer、Open-Door。调度stdout改为 `artifacts/monitoring/r3_Ground-Pick_usd_eval_recovery_20261002_queue.log`，原调度器与单锁未改动，CLoSD正式3072仍未启动。

## 2026-10-02 14:18 UTC（北京时间22:18）：Ground-Pick-Top完成，官方成绩0%

官方1条参考、8192环境、2000迭代、seed42，从零训练至1999，训练和官方默认Fabric路径1000环境500帧视频评估均正常退出0。成功率 **0.0%** 是真实任务失败成绩，不是初始化异常；不重训或重试已正常完成的评估，不宣称效果复现通过。具体原因尚未完成行为诊断。

最终模型SHA256 `8e24d4d55329e1c2e0ccdad5f093d80106fa1748abc8d002c7273cd3642e94a4`；视频 `media/r3_official_ground_pick_top_8192_eval.mp4`，501帧、50fps、10.02秒，SHA256 `929e7d263c2ecd025bc7966b861c49d9a285bab059336c16f94b03162f1a74a5`。模型/日志/配置/视频均独立核验，回执 `results/completed_task_receipts/Ground-Pick-Top_20261002.json`；正式结果 `results/r3_dreamcontrol_official_Ground-Pick-Top_8192_eval.json`。模型仅保留服务器，不随报告上传。

Ground-Pick与Ground-Pick-Top的TensorBoard原始events、全量CSV和曲线已导出本地。前/末100轮平均reward分别−10.108→4.805、−9.759→8.051；末期平均回合约499步；value loss分别0.828871→0.003858、0.737016→0.001828。这支持部分存活/训练奖励改善，但不能代替38.1%/0%官方任务成绩，更不能证明自然抓取与碰撞接触成功。两项均尚无MuJoCo测试。

当前归档 **8/11** 控制任务；原唯一队列已自动进入Button-Press，14:18 UTC核对443/2000轮，随后Open-Drawer、Open-Door。未启动重复队列或修改、中断现有训练。CLoSD新3072正式微调仍等待此前已询问的本地实际场景确认，不重复询问。

## 2026-10-03：Button-Press归档；柜体资产初始化阻塞

Button-Press使用官方96条参考、8192环境、2000迭代、seed42。正常完成1999，官方1000环境500帧默认Fabric视频评估退出0，成功率 **99.9%**。模型SHA256 `e1f83e601e6a45f2aaddc40ffec525677b993bdaa75ed647359ed984d1539670`；本地视频 `media/r3_official_button_press_8192_eval.mp4`，50fps、501帧、10.02秒，SHA256 `aa935bd3ebefa54a21049ec27af7a71c3373e605e4308c10e6922039b87f5b5b`。独立回执 `results/completed_task_receipts/Button-Press_20261003.json`。原始TensorBoard事件与逐轮CSV已导出；前/末100轮平均reward −10.537→6.805，value loss 0.774752→0.000655，回合长度87.5→499.7步。surrogate −0.003514→−0.002023不能单独解释为动作质量变好；尚无本任务MuJoCo迁移验证。

已归档 **9/11控制任务**，不是整篇论文复现完成。Open-Drawer的旧尝试在初始化时失败，没有PPO迭代或模型；原柜体USD只有130字节，是Git LFS占位文件。已恢复官方指针对应17599字节对象，SHA256 `30fddfc98008e380fe37d947e995c7875ead6113544b810d0254dc23450b6a7c`。原始失败记录保全；独立场景/一次更新诊断尚待通过。恢复过程、后续唯一队列启动条件见 `docs/cabinet_asset_recovery_20261003.md`。不把初始化错误记为0%，不重训已完成任务。Open-Door等待。CLoSD新正式微调仍受已有未答复的独立画面确认约束。


2026-10-03 02:31 UTC：独立按文档执行的16环境一次PPO更新诊断已通过，官方柜体SHA与原LFS指针一致。审计确认旧Open-Drawer尝试0次PPO、0个模型，并保留原失败日志哈希；原唯一调度器已恢复，同一tmux/锁跳过已完成七项，余下Open-Drawer→Open-Door，仍为8192环境/2000轮/seed42。当前正式过程处于初始化，不能宣称已完成或已推进；结构化回执见results/r3_dreamcontrol_cabinet_recovery_20261003.json。


2026-10-03 02:33 UTC核实：Open-Drawer run `2026-10-03_02-31-21-Open-Drawer-a` 已写第4/2000轮，日志02:33:28更新；GPU1 18376MiB、73%计算，唯一正式训练进程仍为8192环境/seed42。实际状态帧sampled_env_step=125与最新遥测step130，捕获时间02:33:34，非检查点回放。尚无最终评估成绩。

## 2026-10-03 13:53 UTC：Open-Drawer完成，Open-Door已接续

Open-Drawer官方602条参考、8192环境、2000迭代、seed42，run `2026-10-03_02-31-21-Open-Drawer-a` 正常完成1999；官方1000环境500帧默认Fabric视频评估正常退出0，成功率 **100.0%**。模型SHA256 `026f997e5044708fad6889ca10124edb8fc42218ce031dbdaf423150239bb920`；501帧/50fps/10.02秒视频 `media/r3_official_open_drawer_8192_eval.mp4`，SHA256 `426dcefe74ebd27551177a94c072a44d050aa8382a4a119340514f673f7dc889`。独立核验回执 `results/completed_task_receipts/Open-Drawer_20261003.json`，保留原TensorBoard events、逐轮CSV和曲线。原官方柜体LFS占位文件导致的零更新失败已归档，恢复精确官方资产后才成功；不改PPO、奖励、关节配置或预算。

当前 **10/11控制任务归档**；Open-Door由原唯一队列自动启动，run `2026-10-03_11-04-14-Open-Door-a`，13:53 UTC核实747/2000轮，GPU1持续计算、检查点model_700已保存，实际状态帧step17960→17965有姿态变化，无最终成绩。Open-Drawer尚无MuJoCo迁移或独立未见条件/自然抓握验收；100%仅为官方任务判据结果。CLoSD正式微调仍等待既有本地当前场景确认。

## 2026-10-04 07:27 UTC：全部11项控制任务训练与评估已归档

Open-Door官方602条参考、8192环境、2000迭代、seed42，run `2026-10-03_11-04-14-Open-Door-a` 正常完成1999；官方1000环境500帧默认Fabric视频评估正常退出0，成功率 **99.0%**。模型SHA256 `b70ffda8e6c4f8e922d924570bc90c851e0ff59da8e70fd8e80883f73ddbcdb3`；501帧/50fps/10.02秒视频 `media/r3_official_open_door_8192_eval.mp4`，SHA256 `cc8907490d91a542bfdf2e44501e16c719a4a5954f49c554e470272930af4ba5`。独立核验回执 `results/completed_task_receipts/Open-Door_20261004.json`，保留原TensorBoard events、逐轮CSV和曲线。唯一九任务队列已正常COMPLETE，没有启动额外训练。

累计11/11控制任务有最终结果，不是整篇论文完整复现或所有任务效果通过：Ground-Pick38.1%，Ground-Pick-Top0%仍是已完成的负面结果。只有六任务已有MuJoCo测试，其他五任务无迁移成绩；当前CLoSD正式微调仍待先前实际场景确认。

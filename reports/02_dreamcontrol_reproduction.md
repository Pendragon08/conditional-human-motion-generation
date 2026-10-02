# R3 DreamControl Kick 基线：训练与评估记录

记录日期：2026-09-25，更新至 2026-09-27。**最新状态：官方 66 条参考轨迹、8192 环境、2000 轮的 Kick 基线训练与正式评估已完成，成功率 85.3%。此前 1024 环境基线的 0% 属于不同配置，不能替代这一结果；生成轨迹增强在独立留出集上的收益仍未得到证明。** 上游扩散生成、重定向和修正不因该控制基线成功而自动算作完整复现。本记录服务于新的“通用条件动作生成”课题，不代表旧机器人格斗课题结果。

## 来源与数据

- 官方代码：`GenRobo/DreamControl` commit `ec28b508313d6837de52c68247fbc8cf81fe4935`。训练任务 `Isaac-Motion-Tracking-Kick-v0`，Unitree G1，Isaac Sim 4.5 / modified Isaac Lab。
- 官方提供的 `TrajGen/sample/Kick_sim2` 中 66 条 `.pkl` 参考轨迹经 Git LFS 完整取回；机器人 USD 也按 Git LFS SHA-256 校验。此训练直接使用**官方现成生成轨迹**，尚未复现 `model_humanml` 生成、重新重定向和 refinement 的完整上游流程。
- 官方训练算法、任务、奖励代码未改。为在无桌面服务器提供 WebRTC 直播，对官方 `train.py` 做了一份派生入口，唯一逻辑差异是 Kit 启动后清除追加给 Hydra 的 Kit 命令行参数。构建脚本：`scripts/build_dreamcontrol_livestream_entry.py`；独立环境见证记录：`.aris/compute/dreamcontrol-r3.md`。

## 启动验证

两次 **1-iteration smoke test** 均完成真实 `Learning iteration 0/1`，写入 `model_0.pt`、TensorBoard event、MP4：32 环境和 1024 环境。1024 环境测试第 0 次 mean reward = -2.59，迭代耗时 20.74 秒；这些不是收敛指标。32 环境视频中第 0 帧是初始化黑帧，后续帧能看到多个 G1 与参考关键点；本地样本：`artifacts/videos/r3_dreamcontrol_kick_smoke32.mp4`。

启动过程中修复了独立环境中缺少的 `joblib`、`rich`、`isaac_utils` 路径和反序列化所需 `jaxlie` 等依赖；这些安装均在项目 `/data` 目录下。此前错误启动器可能在 traceback 后返回 0，因此当前以日志中的学习迭代、检查点和视频三项联合判定成功。

## 当前持续训练

| 项目 | 当前配置 |
|---|---|
| 服务器 / GPU | `10.147.17.32`，GPU 0 RTX A6000 |
| 任务 | `Isaac-Motion-Tracking-Kick-v0` |
| 参考数据 | 官方 `Kick_sim2`，66 条 |
| 训练规模 | 1024 parallel environments，2000 iterations，seed 123 |
| 图形查看 | WebRTC livestream 2，NVIDIA 客户端 Server `10.147.17.32`；正式训练中的用户侧可见性待确认 |
| 服务器会话 | `tmux r3_dreamcontrol_kick_formal` |
| 服务器日志 | `artifacts/monitoring/r3_dreamcontrol_kick_formal_1024_2000.log` |
| 模型与事件 | `third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-25_03-56-03-Kick-a/` |
| 本地曲线 | `http://127.0.0.1:6007/`（SSH 隧道，当前训练，不是历史 PBHC 曲线） |
| 本地视频 | `artifacts/videos/r3_dreamcontrol_kick_formal/`；120 秒轮询同步 |

已核实正式训练第 0–9 次迭代连续推进、首个 `model_0.pt` 与 1280×720 的 4.04 秒训练视频产生，视频已同步到本地。GPU 0 运行时显存采样约 6.4 GB。第 0 次 reward -2.59，第 9 次约 -9.46；早期奖励不能代表最终效果或成功率。按当前每迭代约 16–20 秒，2000 迭代粗略需要约 9–12 小时，实际时间以服务器日志为准。训练不因本地客户端断开而自动停止。

本轮训练是采用官方现成参考轨迹的**资源适配基线训练**。与论文默认 8192 环境不同，当前为 1024 环境；因此不得把最终分数直接当成论文原配置的严格数值复现。报告要求的 `model_humanml` 新生成轨迹验证、CLoSD 复现、检查点 play/eval、任务成功率及跨方法比较仍须在后续 Gate 中完成。

## 训练完成后的自动接续（2026-09-25 配置）

服务器上独立的 `tmux r3_dreamcontrol_posttrain_eval` 正等待训练退出标记，不会中断当前进程。它仅在训练正常退出、日志包含第 1999/2000 次迭代、没有 traceback、检查点非空且 GPU 0 已释放时，自动调用官方 DreamControl `play_eval.py` 对最终检查点做 1000 环境评估，输出成功率、评估视频及原始日志。状态文件：`artifacts/monitoring/r3_posttrain_dreamcontrol_eval.state`；日志：`artifacts/monitoring/r3_posttrain_dreamcontrol_eval.log`。任一条件失败则写入 `BLOCKED` 并停止，不能将训练进程退出误判成 Gate 通过。

Codex 当前任务还配置了名为“R3 训练完成后接续实验”的周期跟进。它在训练/评估完成或出现实际阻塞后，继续按 R3 报告的阶段 Gate 推进 CLoSD、基线总结及后续 Ours 实验；状态无变化时不重复启动实验，也不发送无意义的进度提醒。当前自动化配置只代表**已安排接续**，不代表后续阶段已经完成。特别是 CLoSD 环境、HumanML3D 等上游数据仍需按报告逐项核实。

## 并行 GPU 1 中途评估（2026-09-25）

用户允许利用 GPU 1 进行独立任务。保留 GPU 0 的 1024 环境、2000 迭代正式训练不变，将第 800 次迭代检查点复制到隔离目录 `artifacts/r3_interim_gpu1/model_800.pt`，运行官方 `play_eval.py`：物理 GPU 1、128 环境、300 帧、`--headless --video`。这是**中途诊断**，既未修改训练中的模型，也不替代原定的最终 1000 环境、500 帧评估。

评估正常退出，官方脚本打印 `Success rate: 0.0 %`；保存了非空的动作文件、导出策略以及 6.02 秒视频。视频画面可见机器人进行踢腿动作，但小目标触碰成功率仍为零。此值只说明第 800 次迭代检查点在该较短、较小评估设置下未命中目标，不能推断第 2000 次迭代的最终结果。与当前训练和官方正式指标对照时，需固定相同的评估时长及环境数。

- 服务器状态：`artifacts/monitoring/r3_interim_gpu1_model800.state`
- 服务器日志：`artifacts/monitoring/r3_interim_gpu1_model800.log`
- 本地视频：`artifacts/videos/r3_interim_gpu1/eval_a_r3_interim_gpu1-step-0.mp4`
- 可复查启动脚本：`scripts/r3_interim_dreamcontrol_eval_gpu1.sh`

评估完成后 GPU 1 回到约 179 MiB 空闲；GPU 0 的正式训练持续推进。最终 Gate 仍以既定的训练退出标记、最终检查点、正式评估和视频为准。
### 同口径中期曲线队列

已在 GPU 1 启动 `scripts/r3_interim_curve_gpu1.sh`。它只读训练产出的第 800、1200、1600 次迭代检查点，逐个复制到独立目录，以相同的 128 环境、500 帧运行官方评估；每阶段必须产出成功率和非空视频才进入下一阶段。训练过程、最终 GPU 0 评估队列不受改动。第 800 次迭代的同口径评估已完成，成功率仍为 **0.0%**；视频在本地 `artifacts/videos/r3_interim_gpu1/eval_a_r3_curve_800-step-0.mp4`。第 1200 次迭代的同口径评估也已完成，成功率为 **0.0%**，视频在本地 `artifacts/videos/r3_interim_gpu1/eval_a_r3_curve_1200-step-0.mp4`；第 1600 次迭代仍在等待检查点。这些均是中途诊断，不能代替最终评估。状态文件为服务器 `artifacts/monitoring/r3_interim_curve_model{800,1200,1600}.state`。
## 训练曲线访问修复（2026-09-25）

TensorBoard 本身及本地 SSH 隧道正常：`127.0.0.1:6007` 返回 HTTP 200；`/data/runs` 返回当前 run，`Train/mean_reward` 等 Scalar 标签有效，约第 913 次迭代时已有 914 个奖励样本。用户报告当前 in-app `#timeseries` 页面看不到曲线，具体前端原因未能确认。可先打开 `http://127.0.0.1:6007/#scalars` 使用传统 Scalars 视图。

为避免前端渲染问题，另外启动只读实时页面 `http://127.0.0.1:6008/`：它读取同一 TensorBoard Scalar API，绘制奖励、回合长度、目标触碰奖励和超时，每 10 秒刷新，并显示最新迭代与奖励；仅在日志产生新迭代时重绘图像。服务仅监听服务器本机并通过本地 SSH 隧道访问。服务脚本为 `scripts/r3_live_curves_server.py`，tmux 会话 `r3_live_curves`；静态快照 `reports/r3_tensorboard_snapshot.png`，截至第 903 次迭代。该页面只是曲线显示替代入口，不改变训练、日志或评估。
## 训练停滞排查（2026-09-25 10:20 UTC）

正式训练日志和 TensorBoard event 的最后更新时间均为 10:01:20 UTC，最后完整迭代 **1319/2000**，该次平均奖励 -2.82、平均回合长度 488.93、目标触碰奖励 0。到 10:20 UTC 已约 19 分钟未写出新迭代，显著超出之前约 16–20 秒／迭代的正常节奏。进程 PID 2344188 和 tmux 会话仍存在，占用 GPU 0 约 4.5 GiB、GPU 利用率约 30–40%，未产生训练退出标记或 Python traceback；因此状态应记为**停滞待恢复**，不可报告为训练完成，也不能依据旧 ETA 估算剩余时间。

Isaac Sim Kit 日志在同一秒记录 `nvstPushStreamData timeout`、WebRTC stream stop/disconnect，随后 GLFW 图形初始化失败；这是与停滞高度吻合的线索，但尚未以进程栈证明唯一根因。串流服务端口 8011/49100 仍监听。已请用户在本地 NVIDIA 客户端重连作非破坏性恢复验证；当前不向训练进程发送信号，也不修改其配置。

最新完整模型 `model_1300.pt` 已复制为服务器 `artifacts/monitoring/r3_stall_model1300_snapshot.pt`，原件与副本 SHA-256 均为 `1b95d1b5fbcc274662c20462ca2cfa0378aa208d0144346e4acdc38d966286ed`。GPU 1 阶段评估：第 800 与 1200 次检查点在 128 环境、500 帧均为 0.0%；第 1600 次仍等待。训练结束后的正式评估 watcher 仍为 `WAITING_FOR_TRAINING`。
### 停滞期间的检查点独立验收（10:25 UTC）

不接触 GPU 0 进程，GPU 1 使用同一官方评估脚本对第 1300 次迭代检查点执行 128 环境、500 帧测试。检查点可加载，评估正常结束，成功率 **0.0%**；非空视频位于本地 `artifacts/videos/r3_interim_gpu1/eval_a_r3_curve_1300-step-0.mp4`。这证明模型文件可以用于独立推理，但不能证明正式训练已恢复。到 10:25 UTC，主训练日志仍停在 10:01:20 的第 1319 次迭代，训练退出标记未出现，正式后评估仍等待训练完成。

### 11:17 UTC 无损复核与恢复预案

本地 NVIDIA 客户端重连后，Isaac Sim Kit 日志在 10:34:59--10:35 UTC 记录所有 WebRTC stream 再次连接；但截至 11:17 UTC，训练日志与 TensorBoard event 仍没有超过第 **1319/2000** 次迭代，event 文件最后写入仍为 10:01:12 UTC。训练进程 PID 2344188 尚存且占用 GPU，后评估仍为 `WAITING_FOR_TRAINING`，GPU 1 阶段评估第 1600 次仍为 `WAITING_FOR_CHECKPOINT`。因此重连只恢复了串流连接，**没有证据表明训练已恢复**；11:17 UTC 已停滞约 76 分钟。曲线页面的最新点不变是因为训练未写入新数据，不是仅有页面故障。

为避免丢失进度，第 1300 次迭代检查点及独立快照保留不动。已核查派生训练入口支持 `--resume --load_run 2026-09-25_03-56-03-Kick-a --checkpoint model_1300.pt`；其 RSL-RL runner 将 `--max_iterations` 解释为**从已加载迭代起新增的迭代数**，故若获准续训到第 1999 次，应设为 `700`，保持同一任务、1024 环境、seed 123、GPU 0、WebRTC 和视频录制，并写入独立新 run 与日志。随后须以新 run 的最终检查点单独运行正式 1000 环境评估；原后评估 watcher 仍指向旧训练，不可把它的等待状态误报为完成，也不可重复触发旧队列。当前只完成了可恢复性检查，**未停止、重启或修改原训练进程**。此续训属于恢复预案，不是已经取得的结果或 Gate 通过。

### 15:09 UTC 进度复核

训练仍停在第 **1319/2000** 次迭代；TensorBoard event 最后写入时间仍为 10:01:12 UTC，已约 5 小时没有新训练指标。进程与 GPU 占用仍在，但没有正常退出标记。原完成后正式评估继续 `WAITING_FOR_TRAINING`。第 800、1200、1300 次迭代的独立阶段评估均为 0.0%；第 1600 次检查点一直未出现，其 watcher 于 14:30 UTC 写入 `BLOCKED checkpoint_timeout step=1600`。这一状态意味着阶段评估队列因缺少检查点停止，并非第 1600 次评估失败或训练完成。第 1300 次检查点和独立快照仍保存。等待用户对是否停止停滞进程并断点续训的明确答复；此前不改变原训练。

## 用户批准后的断点续训（2026-09-26 00:07 UTC）

用户明确批准停止停滞进程并断点续训。停止前再次核实原训练仍为 **1319/2000**；原 `model_1300.pt` 与独立备份 SHA-256 同为 `1b95d1b5fbcc274662c20462ca2cfa0378aa208d0144346e4acdc38d966286ed`。向旧 Python 进程发送 TERM 后约数秒退出，GPU 0 从约 4.9 GiB 降到约 20 MiB。旧启动包装器写出退出码 `0`，但由于没有第 1999 次迭代，原后评估 watcher 正确写入 `BLOCKED missing_final_iteration`，**未执行原正式评估**；不能将退出码 0 解释为训练完成。

新脚本 `scripts/r3_resume_dreamcontrol_kick_1300.sh` 加载旧 run 的 `model_1300.pt`（RSL-RL 同时恢复策略、优化器及迭代计数），保持任务、1024 环境、seed 123、GPU 0、WebRTC `--livestream=2` 和视频录制不变。RSL-RL 的 `--max_iterations=700` 表示从已加载迭代起再运行 700 次，目标最后为 `Learning iteration 1999/2000`。它写入独立 run `2026-09-26_00-07-40-Kick-a`、独立日志 `artifacts/monitoring/r3_dreamcontrol_kick_resume1300_700_1024.log` 和退出标记 `.exit`，不覆盖旧模型或旧日志。00:09 UTC 已核实加载旧检查点，日志连续写出第 1300、1301、1302 次迭代，新的 TensorBoard event 持续更新；截至 00:09:xx UTC 已超过第 1310 次。此时只是**恢复运行**，最终成功率和 DreamControl Gate 仍未得到。

新的完成后官方评估 watcher `scripts/r3_resume_posttrain_eval.sh` 已启动，状态 `artifacts/monitoring/r3_resume_posttrain_eval.state` 为 `WAITING_FOR_TRAINING`；仅当新 run 第 1999/2000 次迭代、非空检查点和 GPU 0 释放全部核实后，才执行 1000 环境、500 帧官方评估。原第 1600 次中途评估队列已因旧 run 无检查点超时；新 run 使用独立的 `scripts/r3_resume_interim_1600_gpu1.sh` 在 GPU 1 等待新 `model_1600.pt`，届时以与第 800/1200/1300 次相同的 128 环境、500 帧口径评估，状态 `artifacts/monitoring/r3_resume_interim_1600_gpu1.state`。这是并行**评估**而非第二条正式训练。

本地 6007 TensorBoard 原先仍指向旧 run，现已把服务端改指向新 run，并重新建立本地 SSH 隧道。`http://127.0.0.1:6007/#scalars` 可读取新 run 的 `Train/mean_reward`（已见第 1308 次样本），`http://127.0.0.1:6008/` 只读曲线页面也已恢复访问。两个页面均显示新续训；旧 run 记录仍保留在服务器。WebRTC 服务器端口 8011/49100 正在监听，仍需在本地 NVIDIA 客户端重新连接以验收新 run 的机器人实时画面。

00:13 UTC 续训已连续推进至第 **1317/2000** 次迭代，GPU 0 使用约 6.6 GiB，GPU 1 仅有基础占用并等待第 1600 次检查点。新 run 的首个训练视频非空（2,240,422 字节），已同步到本地 `artifacts/videos/r3_dreamcontrol_kick_resume1300/rl-video-step-0.mp4`。该视频只是续训起始片段，不是完成后的正式评估。

## 续训与正式评估完成（2026-09-26 03:38 UTC 核对）

续训正常退出（退出码 0），日志达到 `Learning iteration 1999/2000`，最终检查点为 `third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-26_00-07-40-Kick-a/model_1999.pt`（约 12 MiB）。自动接续的官方 `play_eval.py` 在 GPU 0 上完成 1000 环境、500 帧评估，状态 `artifacts/monitoring/r3_resume_posttrain_eval.state` 为 `COMPLETE`，日志中的 **Success rate: 0.0%**。评估视频 `third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-26_00-07-40-Kick-a/videos/play/eval_a_r3_resume_final-step-0.mp4` 非空（1,744,186 字节）。第 1600 轮在 GPU 1 上的同口径中期评估也完成，128 环境、500 帧成功率为 0.0%。原停滞 run 的后评估状态 `BLOCKED missing_final_iteration` 属于旧 run，不能与续训 run 的已完成正式评估混淆。

本阶段只训练并评估基于官方现成 `TrajGen/sample/Kick_sim2` 的动作跟踪策略；**没有在本项目运行扩散模型新生成任何人体运动轨迹**。训练轮数达到目标，但任务成功率为零，因此不能认定 DreamControl 性能 Gate 通过。后续应先诊断目标触碰与参考轨迹适配，再单独运行并验收上游生成模型。

### 对 0.0% 的只读诊断与流程偏差

DreamControl 官方完整流程是 `model_humanml` 条件扩散生成人体动作（`Kick_sim`）→ G1 动作重定向（`Kick_sim1`）→ 轨迹修正（`Kick_sim2`）→ Isaac Sim/Isaac Lab 的 PPO 控制训练与评估。R3 执行报告第 6.5 节也要求先验收前三步。本次服务器只有官方现成的 66 条 `Kick_sim2/*.pkl` 作为训练输入；`TrajGen/save/` 中没有 `model_humanml`，`Kick_sim/` 没有新生成轨迹。因此这次只完成了官方数据上的**下游控制基线**，并非完整 DreamControl 复现；原计划的先后顺序没有执行完整。

独立只读审计确认：`play_eval.py` 加载 `model_1999.pt`，1000 个环境运行 500 步后以 `100 × 有至少一次命中记录的环境数 / 1000` 计分。Kick 的命中定义是 `right_ankle_roll_link`（右脚踝）在参考踢击时刻前后很短窗口内，与由参考轨迹构成的目标位置距离小于 **0.1 m**；这是仿真中的几何代理指标，不是真实碰撞成功率，也不是动作生成质量分数。本次 `num_envs=1000` 满足官方计数代码的 `<1001` 条件，计数器、奖励注册和检查点加载未发现导致假 0 的代码路径。训练日志第 1300–1999 轮中，700 条 `Episode_Reward/touch_goal` 有 694 条打印为 `0.0000`，仅 6 条为极小非零值，说明正向触达反馈非常稀少。

曾误把 `reset_object_state` 的默认物体高度 0.67 m 当成 Kick 目标高度；Kick 成功判据并不使用此固定高度。已纠正只读脚本 `scripts/r3_dreamcontrol_reference_feasibility.py`：目标来自参考轨迹右脚踝最大前伸位置，66/66 条参考动作在该位置的几何距离均可小于 0.1 m。**不能再声称固定高度使参考动作先天无法得分。** 当前最有依据的解释是策略在严格位置及时序窗口内没有可靠触达目标；但现有日志没有逐环境最小距离、参考轨迹 ID 或时序掩码，尚不能进一步区分跟踪误差、目标对齐误差与评估采样影响。评估仅有一个 checkpoint、一次未固定 seed 的运行，也不能据此概括所有动作和随机种子。读数及哈希见 `reports/R3_DREAMCONTROL_EXPERIMENT_AUDIT.md`。

## 上游条件生成补做（2026-09-26）

为补齐先前未执行的上游，已从 DreamControl `TrajGen/README.md` 指向的官方地址取得 OmniControl `model_humanml3d.pt`，服务器校验后的 SHA-256 为 `886b722e95496f4ceafd704c4e1d0578888733d12ce67429677b80a91a9c36d0`。此外，仓库的 HumanML3D 归一化文件实际是 Git LFS 指针，已按官方 LFS 对象接口逐个下载并用对象 SHA-256 核验；清单为 `artifacts/dreamcontrol-r3/official_dataset_lfs_manifest.json`。按 MDM 官方资源脚本取得授权范围内的 SMPL 运行文件，独立放在服务器的 `TrajGen/body_models/smpl/`，未提交到 Git。

新生成使用 `scripts/r3_dreamcontrol_generate_safe.py/.sh` 调用官方 `sample.generate`。唯一的推理适配是：已有预定义文本和空间提示时，以 HumanML3D 的 `text_only` 加载模式读取相同归一化统计量，避免对未提供的训练集 GloVe 文件产生无关依赖；生成模型和扩散采样方法未改变。首先在独立目录完成 2 条条件踢腿人体轨迹，结果 `artifacts/dreamcontrol-r3/generated/kick_guided_n2_seed10/results.npy`，形状为 `(2, 22, 3, 196)`，控制提示 `(2, 196, 22, 3)`，全部为有限数值，状态 `COMPLETE_VALIDATED`。这验证了**人体动作生成步骤**，尚不等于 G1 轨迹或可执行性验证。

官方预定义 Kick 的 100 条条件批次已在 GPU 0 完成并核验，状态为 `COMPLETE_VALIDATED`，结果 `artifacts/dreamcontrol-r3/generated/kick_guided_n100_seed10/results.npy` 的形状为 `(100, 22, 3, 196)`；提示形状为 `(100, 196, 22, 3)`，两者均无 NaN/Inf，批次间平均运动标准差为 0.1089，文件 SHA-256 为 `b8103eeca19619b10102160b7d74c4dd6c5c05ec0f9c6697ea92e237a1afec45`。100 条动作共用一条踢腿文本，但空间提示不同；因此它验证条件变化下的生成，不能称为 100 种不同文本技能。日志和状态位于 `artifacts/monitoring/r3_dreamcontrol_generate_n100.*`。所有输出写入新的独立目录，未覆盖官方 66 条 `Kick_sim1/Kick_sim2` 样本。

两批新生成动作分别在 `TrajGen/sample_r3_n2/` 与 `sample_r3_n100/` 准备了隔离重定向目录，官方重定向及修正脚本以原样副本放入其中，原始样本不受影响。完整状态见下方预检；正式 RL 训练是否重跑应待新数据质量、图形显示和阶段 Gate 检查后决定，不能把旧 0% 结果算作新数据结果。

### G1 重定向与修正预检

隔离的 PyRoki/JAX GPU 环境已安装并验证 GPU 1 可见；脚本 `scripts/r3_dreamcontrol_prepare_retarget_env.sh` 仅写入 `envs/dreamcontrol-retarget/`，不改动既有训练环境。两条新的人体轨迹经官方 `retarget.py` 和 `refine_motions.py` 原样副本完成 G1 转换：`Kick_sim1` 为 2 条、每条 195 帧、27 关节；`Kick_sim2` 为 2 条、每条 215 帧、27 关节。独立只读验证 `artifacts/monitoring/r3_dreamcontrol_stage_n2_validation.json` 确认两步输出均为有限数值，踢击帧索引有效，URDF 关节限位违反数为 0。**这只证明数据处理管线可执行，不证明新参考轨迹已被策略稳定跟踪或任务成功。**

100 条生成动作的 G1 重定向和轨迹修正现已在 GPU 1 完成，状态、日志与验证结果分别见 `artifacts/monitoring/r3_dreamcontrol_retarget_n100.*`、`r3_dreamcontrol_refine_n100.*` 和 `r3_dreamcontrol_stage_n100_validation.json`。官方重定向筛选保留 **68/100 条（68%）**：6 条因朝向检查被排除，26 条因前移检查被排除。68 条 G1 重定向轨迹各 195 帧，修正后各 215 帧、27 关节；全部数值有限、踢击帧索引有效，按 G1 URDF 检查无关节限位违反。几何复核 `artifacts/monitoring/r3_dreamcontrol_n100_contact_geometry.json` 覆盖 14,620 帧，双脚较低的一侧未出现低于地面 2 cm 的帧，也没有两脚同时高于地面 10 cm 的帧；最低支撑足高度约为 -0.0068 m。该检查仅基于正向运动学，**没有测量接触力、稳定性或实际动力学可执行性**。

由此，R3 第 6.5 节的 `human reference → G1 retargeting → refinement` 数据生成阶段已有一批可复查结果，但 **第 6.7 节完整 DreamControl Gate 仍未通过**：先前 2000 轮 PPO 和 0% 正式成功率均基于官方已有的 66 条 `sample/Kick_sim2`；本轮重生成轨迹尚未经过新的固定预算跟踪训练和任务评估。原提供的 66 条 `sample/Kick_sim1/Kick_sim2` 文件保持原状。

### 与官方现成轨迹的重叠核验

`artifacts/monitoring/r3_dreamcontrol_overlap_audit.json` 对上述 68 条重生成轨迹与官方 66 条 `Kick_sim2` 做只读比对：**65 条文件名相同**，同名轨迹的关节均方根差中位数仅约 **0.0050 rad**、根位置差中位数约 **0.00048 m**；其中 49 条同时低于 0.01 rad 和 1 mm。虽无逐字节完全相同文件，但高度接近，推测主要因同一默认随机种子和重定向数值环境差异。因此，这 68 条是**官方数据生成管线的重跑复现，不是 68 条独立新增示范**。不能将它们简单并入旧 66 条后声称实现数据增强，也不应以旧集对重跑集做独立训练/测试对比。下一步应改用不同随机种子生成独立候选，再按轨迹相似度、物理质量和保留率去重筛选，最后才可在相同训练预算、环境数和评估口径下检验真正的数据增益。

### 不同随机种子的独立候选与质量门（2026-09-26 05:05 UTC）

`seed=42` 的 100 条人体轨迹已完成；与 `seed=10` 逐条对应计算三维关节点 RMSE，中位数 **0.1566 m**，100/100 条均超过 1 cm。结果在 `artifacts/monitoring/r3_dreamcontrol_seed42_human_novelty.json`。这说明采样噪声实际改变了人体轨迹，而不仅是重复写出文件。

随后在隔离目录 `TrajGen/sample_r3_n100_seed42/` 执行官方 G1 重定向，原有 100 条中 65 条通过其内部筛选，4 条因朝向检查排除、31 条因前移检查排除。独立 URDF 限位验收发现 `71.pkl` 的一帧右脚踝滚转关节超限 **0.00154 rad**；它已连同 SHA-256 记录隔离到 `rejected_joint_limit/`，原始文件保留作失败样本，未放宽 0.001 rad 验收容差。其余 **64/100 条（64%）** 经官方轨迹修正脚本生成 G1 参考：每条 215 帧、27 关节，数值有限、踢击帧索引有效、未检出 URDF 关节限位违反。结果和筛选记录见 `artifacts/monitoring/r3_dreamcontrol_stage_n100_seed42_validation.json`、`r3_dreamcontrol_retarget_n100_seed42.log`、`r3_dreamcontrol_seed42_quarantine.sha256`。

跨库最近邻审计 `artifacts/monitoring/r3_dreamcontrol_seed42_crossset_novelty.json` 将每条 seed42 G1 轨迹与官方 66 条参考**全部逐一比较**：最近邻关节 RMSE 中位数 **0.1008 rad**、根位置 RMSE 中位数 **0.1008 m**；同时低于 0.01 rad/1 mm 或 0.02 rad/2 mm 的近重复均为 **0 条**。因此这 64 条是独立候选，区别于上面 seed10 的高度重叠复现。几何审计 `artifacts/monitoring/r3_dreamcontrol_seed42_contact_geometry.json` 覆盖 13,760 帧，未见支撑足低于地面 2 cm 或双脚同时高于地面 10 cm 的帧，最低支撑足高度约 -0.0063 m。**这些是数据与正向运动学检查，尚未验证接触力、动态稳定性或下游策略收益。**

当前可接受的阶段结论是：不同随机种子在已有预训练模型和 100 个 Kick 空间条件下，得到 64 条与官方 66 条不近似重复、形式上满足 G1 轨迹格式及初步质量检查的参考动作。完整 DreamControl Gate 仍未通过，因为尚未对这 64 条进行同预算跟踪训练、任务成功率和跨种子评估；原旧数据策略的正式成功率为 0%，不能代替新数据评估。

### 下游数据收益对照的冻结划分（2026-09-26）

为了检验新增参考轨迹是否改善控制，而不把训练动作拿来做独立测试，`scripts/r3_dreamcontrol_build_comparison_sets.py` 对 64 条 seed42 轨迹按文件名 SHA-256 固定划分为 32 条增强训练、32 条留出测试，并记录每个源文件的哈希。原 66 条保持为基线训练集；增强训练集通过只读链接包含原 66 条加新增 32 条，共 98 条。两组策略后续都必须在同一留出 32 条上，使用相同评估环境数、帧数和成功判据比较。划分清单及目录位于服务器 `third_party/DreamControl/TrajGen/r3_comparison_seed42/manifest.json`。原数据和 seed42 源文件均未移动或修改。

为使官方 Kick 任务读取指定参考集，`scripts/r3_dreamcontrol_select_reference.py` 仅在训练/播放配置的两处 `ref_motions_path` 增加环境变量 `DREAMCONTROL_KICK_REF_PATH`；变量未设置时仍读取官方原目录。原文件哈希和备份保留在服务器，改动后的 Python 语法检查通过。此改动只改变数据入口，不改变策略网络、PPO 奖励或成功判据。

先用旧的 `model_1999.pt` 对全部 64 条新增轨迹做 16 环境、200 帧的读取与播放预检：官方 MotionLib 确认 **Loaded 64 motions**，运行正常结束并产出非空视频；成功率 **0.0%**。它只说明新参考集可被现有仿真接口读取，且旧策略在这一短时口径下未成功，**不是新数据训练效果**。日志和状态为 `artifacts/monitoring/r3_dreamcontrol_seed42_eval_smoke.*`。正式对照必须另行训练增强组，并在固定留出集上评估两组策略。

增强组固定预算启动脚本已准备为 `scripts/r3_dreamcontrol_train_augmented_seed42.sh`，参数为 1024 环境、2000 轮、seed 123、GPU 0、WebRTC 实时画面；它强制要求 `r3_seed42_local_view_gate.json` 记录用户本地实时可见画面，否则拒绝启动。**截至本节记录时未启动新增正式 RL 训练**。这种前置门遵循 `docs/training_visualization_policy_20260917.md`，不能把离线视频或 TensorBoard 曲线替代训练直播。

### 固定留出集上的旧策略基线（2026-09-26）

已用完成的原 66 条策略 `model_1999.pt` 对上述 **32 条独立留出轨迹**运行官方 `play_eval.py`：1000 环境、500 帧、GPU 0、WebRTC 和视频记录。官方 MotionLib 打印 `Loaded 32 motions`，评估正常退出，**Success rate: 0.0%**。日志及状态：`artifacts/monitoring/r3_dreamcontrol_heldout_baseline_eval.*`；非空视频：`third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-26_00-07-40-Kick-a/videos/play/eval_a_r3_seed42_holdout_base66-step-0.mp4`。这给新策略提供同口径基线，但由于基线为零，后续应同时报告更细的动作跟踪误差、接触/目标距离与失败率，不得仅依赖成功率描述数据增益。

已另以旧策略在 **98 条增强训练参考**上启动 16 环境的 WebRTC 画面预检，官方 MotionLib 确认 `Loaded 98 motions`。这只是本地实时可见性和数据接口检查，**没有开展新训练**；是否完成用户侧画面验收以单独记录为准。

首次预检在 30 分钟限定时长后退出，服务器状态为 `STOPPED exit=124`；当时用户尚未确认本地是否看到 G1 画面，图形 Gate 尚待验收。2026-09-26 后续检查发现该预检留下孤立的 `play_eval.py` 进程 PID 2506609，占用 GPU 0/1 显存；核对命令行确认其只是 `r3_seed42_live_view`，已对该进程发送 TERM，端口 8011/49100 释放、两张 GPU 回到空闲，不涉及训练进程或模型。清理记录为 `artifacts/monitoring/r3_seed42_view_cleanup.state`。该时间点增强组正式训练尚未启动；后续启动见下一节。

### 增强组正式训练启动（2026-09-26 16:11 UTC）

用户在重新开启的 16 环境 G1 WebRTC 预检中确认“能看到机器人持续运动”。画面验收记录为 `artifacts/monitoring/r3_seed42_local_view_gate.json`；这是用户侧的实时连接确认，不是从服务器日志推断。预检进程随后单独停止，确认 GPU 空闲后，启动 `scripts/r3_dreamcontrol_augmented_train_eval_queue.sh`。新训练采用原 66 条加 seed42 的 32 条增强轨迹（共 **98 条**）、1024 环境、2000 轮、seed 123、GPU 0 和 WebRTC `--livestream=2`；**未**使用留出的 32 条动作训练。

独立 run 为 `third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-26_16-11-45-Kick-a`，训练状态/日志为 `artifacts/monitoring/r3_dreamcontrol_augmented_seed42_2000_1024.state/.log`。启动后官方 MotionLib 打印 `Loaded 98 motions`，已连续写出至少第 **16/2000** 轮；检查点和正式结果尚未产生，不应预报性能。GPU 0 约使用 6.8 GiB。对应 TensorBoard 服务在服务器本机 6009 端口，经本地 SSH 隧道可打开 `http://127.0.0.1:6009/#scalars`。

同一受锁保护的队列在训练完成、确认第 1999/2000 轮及模型后，自动运行 `scripts/r3_dreamcontrol_augmented_posttrain_eval.sh`：仅在前述 32 条留出动作上，以与旧策略相同的 1000 环境、500 帧口径运行官方 `play_eval.py` 并保存视频。若训练失败或未产生最终检查点，不会把队列状态误报为已评估。旧策略留出集基线为 0.0%；新策略结果目前**未出**。

### 第一次增强训练停滞与断点续训（2026-09-27 01:09 UTC）

第一次增强训练最后完整迭代停在 **119/2000**，日志在 2026-09-26 16:45:57 UTC 之后约 8 小时没有新增；同一时刻出现 `nvstPushStreamData timeout` 和 GLFW 初始化失败。进程仍占用 GPU，但没有继续产生迭代或最终模型。原 `model_100.pt` 与独立备份 `artifacts/monitoring/r3_aug_seed42_stall_model100_snapshot.pt` 的 SHA-256 同为 `e4ea6e574c3a76c45261ef6b28dd8a0b8c8547865e8797578df88705bfaa88d3`，备份 ZIP 内部校验无坏成员。向卡住的训练进程发送 TERM 后 GPU 释放；旧脚本即使得到退出码 0，也正确写入 `BLOCKED`，因为没有达到第 1999 轮。**不能把此 run 说成训练完成。**

根据此前获准的断点续训方案，`scripts/r3_dreamcontrol_augmented_resume100.sh` 从原 run 的 `model_100.pt` 恢复策略与优化器，并以额外 1900 轮为目标继续到全局第 1999 轮。数据仍为相同 98 条、1024 环境、seed 123、GPU 0、WebRTC 实时画面；新 run 与日志独立，未覆盖原 run。恢复 run 为 `third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_01-09-52-Kick-a`，日志和状态在 `artifacts/monitoring/r3_dreamcontrol_augmented_seed42_resume100_1900.*`，已核实加载旧检查点及 98 条参考并写出 **104/2000** 轮。原第 101–119 轮未包含于恢复起点，属于因停滞丢弃的计算，不能计入最终训练预算。

恢复队列 `scripts/r3_dreamcontrol_augmented_resume_eval_queue.sh` 在训练达到第 1999 轮之后自动以相同留出 32 条、1000 环境、500 帧执行官方评估；评估脚本通过明确的恢复 run 日志寻找新最终检查点。TensorBoard 本地 6009 端口已切到恢复 run，`http://127.0.0.1:6009/#scalars` 已验证 HTTP 200。**恢复运行不等于性能 Gate 通过，新评估尚未产生。**

### 训练与 WebRTC 解耦（2026-09-27 01:57 UTC）

第二次含 WebRTC 的续训也在 **207/2000** 停滞：2026-09-27 01:40:55 UTC 训练日志同时出现 `nvstPushStreamData timeout` 与 GLFW 初始化警告，后续迭代不再更新。这是服务器端推流/渲染故障与训练同进程耦合的证据；不能仅凭日志断定本地电脑断网或关机是原因。该 run 的 `model_200.pt` 已复制到 `artifacts/monitoring/r3_aug_seed42_resume_model200_snapshot.pt`，两者 SHA-256 均为 `d1213f419c3ed5ca24a6e9d647e11bb7deb76cee17a0953daa19f026f42904b7`。卡住的旧进程经用户要求修复后收到 TERM，旧队列标记为 `BLOCKED`，没有误触发正式评估；第 201–207 轮不计入后续预算。

新训练队列 `scripts/r3_dreamcontrol_augmented_resume200_queue.sh` 从 `model_200.pt` 恢复，用相同 98 条动作、1024 环境、seed 123、GPU 0 再执行 1800 轮，目标全局第 1999 轮。入口 `scripts/r3_dreamcontrol_augmented_resume200_headless.sh` **不启动 WebRTC、不录制训练视频**，因此直播客户端和编码器故障不再处于 PPO 的计算路径。启动前以 GPU 1、32 环境、1 轮做过一次无串流续训预检，读入同一检查点并实际完成 `Learning iteration 200/201`，退出码 0；这只是启动链路检查，不是正式实验结果。正式 run 为 `third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_01-57-11-Kick-a`，状态与日志为 `artifacts/monitoring/r3_dreamcontrol_augmented_seed42_resume200_1800.*`。在独立画面启动期间训练已继续超过 270 轮；最终成功率仍待训练和留出集评估完成。

GPU 1 上的 `scripts/r3_dreamcontrol_checkpoint_viewer.py` 是**独立的检查点预览进程**：它加载最近保存的模型，在 16 个仿真环境中通过 WebRTC 显示 G1；串流失败时仅重启预览进程，并在发现较新检查点时更新。它**不是正在训练的 1024 个环境的逐帧直播**，因此画面相对于训练有检查点间隔的延迟。预览状态/日志为 `artifacts/monitoring/r3_dreamcontrol_checkpoint_viewer.*`，客户端仍连接 `10.147.17.32`。独立 Kit 使用 `--portable-root` 隔离缓存和配置，首次启动需完成 RTX 着色器编译；在日志出现 `Loaded 98 motions` 之前，连接端口开放也不能视为画面就绪。早期预览重启后遗留的孤立 `play_eval.py` 已按进程组清理，后续 supervisor 明确终止整个进程组。用户侧画面仍待重新确认。

独立预览随后完成场景与 98 条参考动作加载，服务器 8011/49100 端口均处于监听状态。用户在 NVIDIA 客户端重新连接 `10.147.17.32` 后明确确认“能看到机器人运动”。预览曾因初始化诊断过严和退出后孤立子进程而反复重启；现已改为只把实际推流超时视为故障、按进程组清理 Kit，并在画面真正就绪后至少维持五分钟再更新检查点。预览故障排查期间，独立的无串流正式训练从约第 270 轮推进到至少第 951 轮；这是两进程解耦的运行证据。预览会定期加载较新检查点，切换时客户端可能需要重新连接；它仍不是当前训练环境的逐帧直播。

TensorBoard 的 6009 服务已切换到新的 resume200 run，本地 SSH 隧道重新建立并验证 HTTP 200：`http://127.0.0.1:6009/#scalars`。预定的正式留出评估入口 `scripts/r3_dreamcontrol_augmented_posttrain_eval.sh` 已改为 `--headless --video`，保留原 1000 环境、500 帧和官方成功判据，在训练完成后离屏录制视频；因此正式评估也不再把 WebRTC 放在计算路径。此修改不改变训练、预览和正式留出评估之间的指标边界。最终训练和评估仍未完成，不能预报成功率。

### 按论文训练预算复现官方 Kick 基线（2026-09-27 02:36 UTC 排队）

原 66 条 Kick 参考的已完成基线使用 1024 个环境、2000 轮，成功率 0%；增强组沿用 1024 环境，二者适合在固定预算下比较数据增益，**均不是论文的严格数值复现**。[DreamControl 论文原文](https://genrobo.github.io/DreamControl/static/paper/DreamControl_arxiv.pdf)明确写每个任务在一张 RTX A6000 上以 PPO、**8192 并行环境、2000 迭代**训练；表 II 报告 Precise Kick 成功率 **98.6%**。这个发表数值不是本地结果，也不能承诺本地一定达到。

已核对官方 `G1KickEnvCfg.scene.num_envs=8192`、Kick 的 `G1FlatPPORunnerCfg`、官方 66 条 `Kick_sim2` 文件及原训练入口。当前 Kick 配置仅在两处增加可选数据路径；当 `DREAMCONTROL_KICK_REF_PATH` 未设置时，仍读取官方 `../TrajGen/sample/Kick_sim2`。论文未在该段指定随机种子，使用代码基类默认 seed 42 并显式记录；由于官方 `train.py` 的命令行默认迭代数为 5000，必须显式传 `--max_iterations=2000` 才与论文一致。训练沿用官方 README 的 `--headless` 路径，不让 WebRTC 干扰 PPO；本地能看的独立检查点预览与训练数值分开报告。

脚本 `scripts/r3_dreamcontrol_official_kick_8192.sh` 以官方 `train.py` 从零训练、原 66 条参考、8192 环境、2000 轮、seed 42、GPU 0；`scripts/r3_dreamcontrol_official_kick_8192_eval.sh` 以官方 `play_eval.py` 在 1000 环境、500 帧执行正式成功率评估并保存视频。队列 `scripts/r3_dreamcontrol_official_kick_8192_queue.sh` 已在 tmux `r3_dreamcontrol_official_kick_8192` 启动，当前状态 **QUEUED**：先等待已经运行的 98 条增强组训练及其独立留出集评估退出，再确认 GPU 0 空闲，随后自动开始官方 66 条基线，不会打断或覆盖已有 run。状态文件为 `artifacts/monitoring/r3_dreamcontrol_official_kick_8192_queue.state`、`r3_dreamcontrol_official_kick_8192_2000.state`、`r3_dreamcontrol_official_kick_8192_eval.state`。截至排队时尚未得到这一论文预算复现的训练或评估结果。

### 增强组 2000 轮与留出评估完成（2026-09-27 03:11 UTC）

从 `model_200.pt` 解耦 WebRTC 后的训练达到全局 **1999/2000** 轮，状态 `r3_dreamcontrol_augmented_seed42_resume200_1800.state=COMPLETE`；最终检查点 `third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_01-57-11-Kick-a/model_1999.pt` 大小约 12 MiB。既定队列随后在固定的 32 条独立留出参考上以 **1000 环境、500 帧**运行官方 `play_eval.py`，日志打印 `Success rate: 0.0 %`，状态 `r3_dreamcontrol_augmented_seed42_heldout_eval.state=COMPLETE`；视频 `.../videos/play/eval_a_r3_aug_seed42_holdout-step-0.mp4` 非空（1,665,313 字节）。旧 66 条策略在相同留出集上的成功率也是 0.0%。因此**本次生成轨迹增强并未在该成功率指标上产生可证实收益**；不能把完成训练等同于数据增强有效。后续需先检查动作跟踪误差、失败原因和论文参数基线，再决定是否调整生成数据或控制训练。原 66 条、8192 环境的论文预算队列仍单独排队，尚未得出结果。

原论文预算队列随后发现等待条件中的 `r3_dreamcontrol_aug_seed42_resume200` 会话仍存在，但该会话实际只承载 TensorBoard，正式训练和留出评估均已完成。将该 TensorBoard 会话**仅重命名**为 `r3_tensorboard_resume200`，没有停止服务或更改任何训练进程；原队列遂自行进入 `RUNNING official66_8192_2000`，对应训练状态记录原 66 条、8192 环境、2000 轮、seed 42、GPU 0。此处仅证明论文参数基线已启动，结果仍待完成。

首次启动在训练迭代前因服务器缺少 `git-lfs`，Isaac Lab 记录 Git 差异时报错。原状态和日志已另存为 `r3_dreamcontrol_official_kick_8192_attempt1_git_lfs_missing.*`。从 Git LFS 官方发布页下载 Linux amd64 v3.8.0，SHA-256 核对为 `e455e00f15d9b95661b8d53498ffb0c3367962cf1ec73c31ab7369516cd6ab8d`，仅放在项目 `cache/git-lfs-3.8.0/`，并仅向本次基线脚本的 `PATH` 添加该工具。预先运行同一 Git diff 命令已无错误；旧队列退出后以原参数重新启动，日志已连续写出至少 **7/2000** 轮。修复只影响代码记录依赖，不改变 66 条参考、训练算法或环境数量；正式评估仍待训练结束。

### 官方参数 Kick 基线完成（2026-09-27 05:44 UTC 核对）

原 66 条参考动作、8192 个并行环境、2000 轮、seed 42 的独立基线已完成至 **1999/2000**，最终检查点为 `third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a/model_1999.pt`（12,565,922 字节）。队列随即以官方 `play_eval.py` 在 **1000 环境、500 帧**的口径评估，日志报告 **Success rate: 85.30000448226929%**。状态文件 `artifacts/monitoring/r3_dreamcontrol_official_kick_8192_2000.state`、`r3_dreamcontrol_official_kick_8192_eval.state` 和队列状态均为 `COMPLETE`。评估视频 `third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a/videos/play/eval_a_r3_official_kick_8192_eval-step-0.mp4` 非空（1,613,524 字节）。这是**官方 66 条训练动作上的基线评估**，与论文表 II 的 98.6% 尚有差距；seed、硬件、代码版本和评估细节均应记录后再分析差异。

先前 1024 环境的 66 条基线与 98 条增强组在另一个固定的 32 条新增动作留出集上都得到 0%。**85.3% 不能与那两个 0% 直接作为数据增强收益对比**：训练环境数、随机种子及评估动作集合不同。下一步应使用本次 8192 环境基线检查点在同一 32 条留出集上补做评估，再以匹配的环境数和训练预算重训增强组，才能判断生成数据是否改善泛化。

## 2026-10-02：第八项控制任务归档

Ground-Pick-Top官方1条参考、8192环境2000轮seed42训练及1000环境500帧官方默认Fabric评估均正常退出0，成功率0.0%。已核验1999模型及501帧10.02秒视频，原始指标与核验回执已保存。该结果不表示效果复现通过，也不记为初始化故障。当前8/11控制任务已有正式归档，Button-Press仍在唯一队列训练；详见[多任务时间线](03_dreamcontrol_multitask_baselines.md)。

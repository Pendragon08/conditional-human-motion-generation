# R3 CLoSD 基线：资产、环境、推理与训练状态

更新：2026-09-28。**512 环境缩容微调已达到目标 epoch 66000 并保存检查点；固定检查点的独立多任务评估已完成，但论文 4096 环境指标 Gate 尚未通过。** 训练进程在最终保存之后以退出码 139 结束，不能将该退出码或训练期间的即时成功率当作正式评估结论。下述早期 GPU witness 和命令行检查仅记录准备过程。

## 官方来源与资源

- 官方代码：`GuyTevet/CLoSD`，服务器 `third_party/CLoSD/`，固定 commit `d8950bdb78312fcd2844f336b3fc079e7368c87f`；平台 Isaac Gym Preview 4。
- 官方 9 环境推理预计约 4 GiB 显存；官方 4096 环境评估／训练写为约 50 GiB，单张 A6000 只有 49,140 MiB 总显存。未经容量实验，不能声称已完整复现官方 4096 环境设置。
- 官方 Hugging Face 模型库 `guytevet/CLoSD` 固定修订 `de7106b947b6f70700b5320d1cd61fef4a9ebc9b`。用户于 2026-09-26 确认已取得 SMPL、AMASS、HumanML3D 各自许可并授权下载。本地电脑不保存这些受许可资产，也不提交到 Git。
- 服务器无法直接连接 Hugging Face。使用本地到服务器的私有反向代理（仅服务器 `127.0.0.1:18181`），官方 `snapshot_download` 正把完整资产缓存在服务器 `/data` 盘的 `cache/closd_hf/`。进度见 `artifacts/monitoring/r3_closd_asset_download.state` 和 `.log`；只有生成 `artifacts/closd-r3/official_assets_manifest.json` 并通过必需文件核验后，资产 Gate 才算通过。目录容量增长不是下载完成证据。

## 独立环境与预检

- `env-closd-r3.json` 是声明式依赖及资产规格，`scripts/prepare_closd_r3_env.sh` 把依赖安装在 `envs/closd-r3/site-packages/`；原 `.venv-pbhc-training` 的 Python 3.8.20、Torch 1.13.1+cu117、Isaac Gym 保持冻结。安装使用 `--no-deps`，避免上游依赖把 Torch 替换为 2.4。
- CLoSD 实际只从 PyTorch3D 使用 `euler_angles_to_matrix` 和 `matrix_to_quaternion`。官方 `pytorch3d` v0.3.0 源码中的纯 Python `rotation_conversions.py` 与 BSD-3-Clause 许可证由 `scripts/stage_closd_pytorch3d_rotations.sh` 放入独立环境；没有修改 CLoSD 算法代码或编译 PyTorch3D 其他模块。
- 官方 Isaac Gym `gymtorch` 扩展已在独立缓存中成功编译。`scripts/r3_closd_cli_preflight.sh` 的 `env=closd_multitask --help` 已成功解析，实际官方多任务配置的 `learning.params.config.player.games_num=9` 也通过 Hydra `--cfg job` 检查。最初不带 `env=` 的 `--help` 因官方 `config.yaml` 默认引用仓库不存在的 `env_im` 而失败；正式命令应按 README 显式指定 `env=closd_multitask`。
- 种子固定的 GPU kernel witness 已运行：`CLOSD_R3_GPU_WITNESS (8, 8) (1, 4) NVIDIA RTX A6000 97.6353 0.980066`。这验证 Torch 矩阵计算与官方 PyTorch3D 姿态转换在 GPU 1 执行，不等于 CLoSD 模型加载或仿真任务已成功。
- 官方 CLoSD 使用的 OpenAI CLIP `ViT-B/32` 权重已按原库加载，并通过一条文本编码：`CLOSD_R3_CLIP_WITNESS (1, 512) 9.557226`；日志为 `artifacts/monitoring/r3_closd_clip_witness.log`。这仍不是 CLoSD planner 的输出。

## 后续阶段 Gate

1. 完整官方快照下载结束并核对 SMPL、AMASS、HumanML3D、控制器与扩散模型检查点；记录文件数量、大小与修订。
2. 避开 GPU 1 上已排队的 DreamControl 第 1600 轮评估，运行限定轮数的 CLoSD 官方 9 环境多任务推理，记录 planner 输出、控制器执行和实际 GPU 占用。单独标明若采用无界面诊断推理，它不是用户本地实时画面验收。
3. 使用官方可用的评估命令和与 A6000 适配的环境规模，记录与论文表格是否可比；缩容结果不得冒充官方 4096 环境数值。
4. 启动新的 CLoSD 仿真 RL 训练前，按 `AGENTS.md` 和 `docs/training_visualization_policy_20260917.md`，先验收本地可见的实时 Isaac Gym 图形画面，以及曲线、录像保存。无法通过此 Gate 时不悄悄转为无界面训练。

`scripts/r3_closd_multitask_inference_queue.sh` 已在服务器 `r3_closd_multitask_inference` 会话排队：完整快照和 DreamControl 第 1600 轮 GPU 1 评估结束后，才尝试一次有上限的官方 9 环境多任务推理；执行结果会写入独立 state/log，退出码为 0 仍需审查 planner 与控制器证据。它不启动正式训练。

图形训练预检：服务器当前 Xorg `:0` 为 LightDM 登录界面，SSH 用户没有认证 cookie；直接启动新 Xorg 被拒绝访问 `/dev/tty0`。项目内临时运行的 Xvfb `:96` 可以连接，但 `glxinfo` 显示 Mesa `llvmpipe`、`Accelerated: no`，不符合 GPU 图形渲染验收，已停止。因而正式 CLoSD RL 训练的实时图形 Gate 目前**未通过**，不能把上述无界面推理或 Xvfb 预检说成用户可见的训练画面。

上文为启动时的计划快照；后续实际结果见下方时间线。

## 官方资产完成与首次推理阻塞（2026-09-26 03:38 UTC 核对）

授权范围内的官方 `guytevet/CLoSD` 快照已完整下载至服务器 `/data` 缓存。`artifacts/closd-r3/official_assets_manifest.json` 记录固定修订 `de7106b947b6f70700b5320d1cd61fef4a9ebc9b`、52 个文件、总计 19,281,855,442 字节，并核验了 SMPL、AMASS、HumanML3D 与模型检查点等必需文件。下载期间私有代理曾中断，恢复后利用缓存续传；最终资产状态为 `COMPLETE`。这仅表示资产 Gate 通过。

原定的 GPU 1、9 环境多任务推理已启动，官方代码加载 `CLoSD_multitask_finetune/Humanoid.pth` 并进入 GPU PhysX 初始化，但日志停在动作采样预处理进度 `0/8`；20 分钟限定时间到达后退出码 124，状态为 `BLOCKED inference_exit=124`。目前**没有可验收的 planner 输出、控制器执行结果或新生成的人体运动轨迹**，不得把模型加载与仿真初始化算作成功推理。正式 CLoSD RL 训练仍未启动，实时 GPU 图形显示 Gate 也未通过。下一步先诊断动作数据加载停顿，再决定是否有依据地调整推理超时或资源设置；不能直接声称 CLoSD 复现完成。

## 串行加载诊断推理（2026-09-26 04:16 UTC）

原 9 环境官方运行卡在动作库 8 个 worker 的加载队列。独立诊断脚本 `scripts/r3_closd_multitask_inference_serial_diagnostic.sh` 保持官方 CLoSD 模型与仿真代码，设置 `debug=True` 使动作加载改为单进程，并仍使用 GPU 1、9 个环境。首次运行在缺少 `distilbert/distilbert-base-uncased` 文本编码器缓存处停止。随后经私有代理补齐该公开编码器并写入 `artifacts/closd-r3/text_encoder_manifest.json`，第二次诊断正常退出：日志显示 `Loaded 9 motions`、扩散规划器与已训练控制器检查点加载，以及 299 步的仿真回合；打印 `av reward: 108.39787462022569`、`av steps: 299.0`。原始状态与日志为 `artifacts/monitoring/r3_closd_multitask_serial_diagnostic.state/.log`。

这是**缩容诊断推理**，且 `debug=True` 改变了动作库加载方式，不能冒充原始并行配置或论文 4096 环境指标。当前日志没有逐任务成功率、完整 planner 轨迹导出或本地实时 GPU 图形画面，因此不能据此判定论文 Phase B Gate 或正式 RL 训练 Gate 通过。下一步需单独核验规划器产生的轨迹、各任务结果与实时可视化，再决定是否进行正式训练。

## 2026-09-27 进度复核

资产下载状态仍为 `COMPLETE`；首次官方 9 环境并行推理为 `BLOCKED inference_exit=124`，因为动作库加载在 20 分钟内未完成。随后 `debug=True` 的单进程加载诊断状态为 `FINISHED_NEEDS_VALIDATION`，完成 299 步仿真并显示平均回报约 108.40，但没有逐任务成功率或规划轨迹导出。`artifacts/monitoring/r3_closd_training.state` 仍明确为 `BLOCKED ... gpu_gui_gate_pending; no_formal_rl_training`。因此 CLoSD **没有跑完正式训练，也没有完成论文口径的多任务评估**；诊断退出码 0 不能替代这两项结果。接下来应先获得逐任务/规划器输出与 GPU 图形显示证据，再决定正式 RL 的环境规模和运行时间。

## 原论文流程与离线数据增强实验的边界

[CLoSD 正式论文（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/file/7319b7561ffe5e2f6419acd4a2f52d6b-Paper-Conference.pdf)中的 DiP 在 HumanML3D 文本—人体动作数据上训练，输入运动前缀、文本与目标关节位置，连续生成短期运动计划；单一 PHC 跟踪策略先在 AMASS 动捕库上预训练。闭环微调时**固定 DiP**，在 Isaac Gym 内由 DiP 在线生成参考，策略逐帧执行，仿真运动再作为下一次规划前缀，并以 PPO 微调跟踪策略。论文记载 PHC 在 AMASS 上预训约 62K PPO epochs、闭环再微调约 4K epochs，闭环训练使用单张 A100 与 3072 并行环境。论文方法**不是**先将 DiP 离线生成动作与 AMASS 文件混合成一个数据集，再从零训练策略。

如果本课题研究“生成数据是否改善控制训练”，可以另设**衍生的数据增强实验**：冻结同一 DiP 与 PHC 初始化，导出可复查的文本/目标条件轨迹，分别用原始动作集和原始+新增轨迹集在相同预算下训练/评估，再与官方在线闭环方案对照。必须记录生成数据与原 AMASS/HumanML3D 的重叠、质量、物理执行及保留测试集；该实验不能写成 CLoSD 官方数值复现。当前服务器仅完成缩容诊断推理，尚未完成这项离线轨迹导出、混合数据训练或正式闭环 PPO 微调。

## 2026-09-27：独立实时监控预检

“9 环境”是 **9 个并行的 Isaac Gym 仿真实例**，不是 9 种任务。官方多任务初始化选项包含踢击、拳击、坐下、触及目标，坐下后可进入起身状态；初始化选项中的坐下与触及目标各重复两次。任务类别与环境数量不能混为一谈。

GPU 0 是 DreamControl 的 1024 环境增强组续训；GPU 1 同时运行 DreamControl 独立检查点预览与 CLoSD 的 9 环境多任务推理预检。预检时两张 A6000 各有约 44 GiB 与 39 GiB 空余显存，说明**当前 9 环境推理可并行**，但不证明 512 环境微调或论文 3072 环境配置必然能装入显存。正式训练前需独立容量检查。

新增 `scripts/r3_closd_live_dashboard.py` 作为独立监控进程，仅接收由 CLoSD 仿真实时状态发出的非阻塞 UDP 抽样。网页 `http://127.0.0.1:6011/` 同时显示 DiP 生成的下一帧目标骨架、Isaac Gym 已执行的骨架与 DreamControl 迭代；关闭网页、SSH 隧道或监控进程不会使训练等待网络。`scripts/r3_closd_visual_preflight.sh` 当前只运行 9 环境**推理**，其状态/日志为 `artifacts/monitoring/r3_closd_visual_preflight.state/.log`。实测网页收到连续姿态包，且 DreamControl 迭代仍推进。网页显示的是实际仿真数据的骨架视图，**不是 Isaac Gym 原生全场景 GPU Viewer**，也不是正式 RL 训练画面，故不能冒充原有图形 Gate 已通过。

正式 CLoSD 训练启动器 `scripts/r3_closd_live_train_512.sh` 已准备为 512 环境的缩容微调候选，训练与浏览器监控各为独立进程；它目前**未启动**，因为项目的本地实时 GPU 图形训练 Gate 尚未完成，也没有 512 环境容量预检。启动器要求记录用户明确接受“实时骨架监控替代原生 Viewer”的选择；仅看到推理预检页面不自动满足此条件。512 环境不是论文的 3072 环境严格复现。网页推理预检结果不构成任务成功率或训练效果结论。

512 环境的**非训练容量测试**已单独排队在 `r3_closd_capacity_512` 会话：仅待 9 环境可视化预检结束后运行，限定短回合与 15 分钟，不更新策略。状态与日志分别写入 `artifacts/monitoring/r3_closd_capacity_512_inference.state/.log`。排队不是容量通过；正式微调仍需容量测试及图形 Gate 的结果。

## 2026-09-27：Isaac Gym 全场景画面与正式微调启动

- 先前 6011 页显示“数据已暂停”，原因是 9 环境骨架推理预检在预设时限结束后停止发送 UDP；这不代表 DreamControl 训练停下。该页现增加全场景入口，并优先读取新的 CLoSD 正式微调状态。完整三维场景入口是本机独立 SSH 隧道内的 http://127.0.0.1:7001/static/（映射到服务器 7000 端口）。
- 接入 NVIDIA sim-web-visualizer（仓库提交 20b02f3380b7872bf262ac26859879d1c1385671）和其 MeshCat fork，CLoSD 在 opt-in 环境变量下使用 headless=False，将 Isaac Gym 地面、SMPL 角色和场景物体绘制到本地浏览器。仅展示最前面的两个仿真环境，物理仿真仍在服务器 GPU 1；这不是服务器原生 X11 Viewer 或训练结果评估。
- 9 环境全场景推理预检中，用户明确确认“能看到机器人和场景持续运动”。画面服务与仿真是两个进程。人为关闭独立 MeshCat 服务后，日志记录 R3_CLOSD_SCENE_MONITOR_FAILED_SIM_CONTINUES，仿真在首次 299 步评价后又完成了至少三轮评价；故页面或画面服务故障不再必然中断仿真。预检由操作员主动结束，state 中的 exit=143 不是算法训练失败。
- 512 环境短程推理容量测试已完成，状态为 COMPLETE inference_capacity_only；这仅说明短程推理可装入 GPU 1，不代表论文 3072 环境配置可用。正式微调使用 512 环境、官方 CLoSD_no_finetune 检查点、DiP 模型和独立网页画面，不得与官方论文数值直接比较。
- 首次正式微调启动器把 max_epochs=4000 误设为总轮数，而源检查点已有 epoch=62000，因此只执行一次更新就触发 MAX EPOCHS NUM，进程随后以 139 退出。该次状态、日志和模型另存为 failed_v1，不计为有效训练结果。修正后的独立 v2 从源检查点读取基准 epoch=62000，设置目标总 epoch=66000，即追加 4000 轮；2026-09-27 03:56 UTC 已启动。运行状态在 artifacts/monitoring/r3_closd_full_scene_train_512.state，日志同名 .log，输出目录为 third_party/CLoSD/output/CLoSD/r3_CLoSD_multitask_finetune_512_full_scene_v2。启动不等于训练 Gate 或指标已通过，需待实际迭代、检查点和任务评价核实。
- v2 已越过准备阶段，日志记录 epoch 62004、累计 65,536 帧、当轮平均奖励 43.8、即时成功率 0.00；这是早期训练曲线，不能推断最终成绩。GPU 1 显存约 21.5 GiB；6011 页重新收到真实步骤 UDP 数据，不再显示“数据已暂停”。GPU 0 上 DreamControl 官方 66 动作、8192 环境复现仍独立运行。首次全场景预检截图及用户确认只验收可见性，不替代正式成功率测试。
- 画面服务故障隔离采取 fail-open：服务崩溃时当前进程停止向该服务发送绘图命令，物理仿真和训练继续。当前版本尚不能在同一次训练中自动重建完整场景并热重连画面服务；关闭或重开浏览器/本地隧道、但服务器画面服务仍运行时，可重新连接。该限制不应被写成“直播自动恢复”。

### 2026-09-27：正式训练画面中断与独立检查点预览

正式 512 环境训练在约 epoch 62110 的画面发送中出现 `R3_CLOSD_SCENE_MONITOR_FAILED_SIM_CONTINUES: Again('Resource temporarily unavailable')`。fail-open 钩子随即关闭该进程后续绘图请求；MeshCat 网页与本地 SSH 隧道仍可连接，但显示的训练场景变为旧帧。训练日志此后继续超过 epoch 62230，故**网页可打开不代表实时训练画面仍在更新**，而画面故障也未阻断 GPU 训练。本次运行不能只靠重连浏览器、重启网页服务或切换视频协议重新获取已停止发送的训练帧；不为修复画面而中断正式训练。

作为补充观察，`scripts/r3_closd_checkpoint_preview.sh` 在独立进程和 GPU 0 上复制最近完整的正式训练检查点，用官方 CLoSD 多任务推理及 Isaac Gym/浏览器场景展示动作，不修改 GPU 1 的正式训练。首次快照为 epoch 62200；状态和日志在 `artifacts/monitoring/r3_closd_checkpoint_preview.*`。本地 `http://127.0.0.1:7001/static/` 已重新显示可变化的机器人画面。`scripts/r3_closd_checkpoint_preview_supervisor.sh` 等待首次预览结束后负责短时故障重试和定期加载更新的检查点。**这只是带有检查点延迟的独立演示，不是正在训练的 512 环境逐帧直播，也不构成新的性能评估。** 原 DreamControl 的 NVIDIA WebRTC 串流属于 Isaac Sim/Kit，不能直接绑定到 CLoSD 的 Isaac Gym 场景；若要 WebRTC，需要另行建立独立的视频编码与信令桥，不能把换协议说成恢复了原训练帧。

### 2026-09-28：目标检查点已保存，正式评估独立进行

- v2 日志已记录 epoch 66000 和 66001，且 `Humanoid_00066000.pth` 在进程退出前保存成功；文件大小 216,577,317 字节，SHA-256 为 `4c3eb7fa77b0a29743ff21d25fc65ba3d9f2b40d9e9b1e499289707b25add5df`。退出时出现 `MAX EPOCHS NUM!`，随后进程以 139 退出，状态文件仍如实保留为 `FAILED train_exit=139`。这说明目标轮次与检查点已达到，但正常退出验收未通过；不得把它当成完整论文复现。
- `scripts/r3_closd_final_eval_512.sh` 从该固定检查点复制评估专用副本，使用官方 `closd_eval=True` 多任务统计逻辑，在 GPU 1 上独立评估。评估设置为 512 环境、单回合 500 帧、DiP CFG 7.5；官方论文表 1 使用 4096 环境，因此本次是显存受限的缩容评估，不能直接宣称论文同配置复现。状态与逐任务输出分别在 `artifacts/monitoring/r3_closd_final_eval_512.state` 和 `.log`，二者已记录正常完成及 `Eval plot ended`。
- 官方评估器累计每个状态至少 1000 次重置后，输出以下**四舍五入到小数点后两位**的成功率：REACH 0.00、STRIKE_KICK 0.05、STRIKE_PUNCH 0.05、SIT 0.07、GET_UP 0.69。结果结构化保存在 `results/r3_closd_finetuned_eval_512.json`。官方实现将踢击与出拳放入同一组 strike 计数器，两者相同的 0.05 **不是两个独立估计**；也不应把这五列简单平均成“总体成功率”。
- `scripts/r3_closd_pretrained_eval_512.sh` 已在同一 512 环境、500 帧、CFG 7.5、seed 0 口径下完成官方 `CLoSD_no_finetune` 检查点的独立评估；日志同样出现 `Collected enough data (1000 per task)` 与 `Eval plot ended`。源检查点 SHA-256 为 `a47bc0716ce0c40ffdbf1b093952e07182d47bf645123852df05c8c91faa68c2`。对照结果保存在 `results/r3_closd_pretrained_eval_512.json`：

  | 官方评估状态 | 未微调检查点 | 512 环境微调至 epoch 66000 | 变化 |
  | --- | ---: | ---: | ---: |
  | REACH | 1.00 | 0.00 | -1.00 |
  | STRIKE_KICK / STRIKE_PUNCH（官方共用计数） | 0.87 | 0.05 | -0.82 |
  | SIT | 0.77 | 0.07 | -0.70 |
  | GET_UP | 0.56 | 0.69 | +0.13 |

  在这一**单种子、缩容**对照中，微调严重降低触达、攻击和坐下成功率，仅起身提高；因此不能声称本次 4000 轮微调有效。这里的变化是按日志四舍五入后的差值，不是带置信区间的总体结论。需进一步排查缩容训练配置、任务采样与权重更新，之后再决定是否复训；不能用训练即时 `succ_rate` 推翻固定检查点评估。受许可约束的原始数据和模型权重保留在服务器，不上传 GitHub；仓库只保存脚本、配置、哈希和经核实的指标。

  训练日志的**单轮即时统计**也提示后期可能退化：epoch 62050 为 0.91，62500 为 0.48，65000 为 0.56，65500 为 0.26，65900 为 0.00；这些值受任务混合与当前批次影响，不能直接等同上述每状态 1000 次重置的测试率。当前应将该 512 环境微调标为失败的性能改进尝试，保留官方未微调模型作为强基线，并先诊断任务采样和训练配置，避免继续扩展失效配置。

### 初步根因审计：架构相同，不是同配置复现

- 官方 [论文表 1](https://proceedings.iclr.cc/paper_files/paper/2025/file/7319b7561ffe5e2f6419acd4a2f52d6b-Paper-Conference.pdf) 中，未微调的闭环模型在触达／攻击／坐下／起身上的成功率为 1.00／0.81／0.66／0.53，微调后为 1.00／0.90／0.88／0.98。我们的未微调检查点在相同任务顺序上得到 1.00／0.87／0.77／0.56，接近论文未微调水平；因此首先需要解释的是**本次微调为何退化**，而不是笼统认定扩散规划器或评估器完全失效。不同环境数量和随机性仍限制论文数值的直接比较。
- 服务器代码固定在官方提交 `d8950bdb78312fcd2844f336b3fc079e7368c87f`。`git diff` 显示四处修改：`closd.py` 增加只读遥测；`humanoid.py` 和 `humanoid_amp.py` 增加 MeshCat 图形显示与故障隔离；`motion_lib_base.py` 允许串行加载动作。没有修改 DiP/PHC 网络结构、PPO 损失或奖励的代码。训练日志确认加载官方 `CLoSD_no_finetune/Humanoid.pth`，评估日志确认加载官方 DiP 权重。**主体架构相同，但仿真可视化包装层并非未经改动的官方代码**；其潜在运行影响尚未通过消融排除。
- 按[官方 README](https://github.com/GuyTevet/CLoSD#train-your-own-closd)，论文多任务微调为 3072 个并行环境、4000 个 PPO epoch；本次为 512 个环境、同样 4000 epoch。官方 `im_big.yaml` 使用 32 步 rollout、16,384 的 minibatch；因此本次每轮收集 `512×32=16,384` 帧，官方配置每轮为 `3072×32=98,304` 帧，总交互样本量约少六倍，每轮的任务覆盖和 PPO 小批次数也不同。这是**明确的配置差异与合理的退化假设**，但还不能据此断言它是唯一原因。论文评估另采用 4096 环境，本报告的两组匹配对照都只用 512 环境。
- 下一步按证据优先级做只读／小规模诊断：核对 62000、62500、65000、66000 检查点的同口径成功率以定位退化起点；审查训练任务分布和 PPO 更新统计；再用固定检查点比较有无可视化包装的评估，排除包装层影响。完成前不要继续使用当前 512 环境配置训练，也不要把它的失败归因于 CLoSD 原论文架构。

## 2026-10-01：官方规模复现恢复队列

用户要求继续复现。进一步核对官方未微调权重、已发布微调权重和旧 512 环境微调权重中的优化器学习率，均为 2e-5，未发现此前猜测的学习率错误。旧日志的退出码139发生于最终保存及 `MAX EPOCHS NUM!` 之后，原日志缺少原生栈；因此只能说“疑似原生库退出清理故障”，不能声称已定位到具体调用、显存溢出或算法结构。退出异常与固定模型评估退化是两个问题。

在新通用动作项目下建立固定官方提交 `d8950bdb78312fcd2844f336b3fc079e7368c87f` 的干净代码副本，复用已取得许可的资源、Python/Torch/Isaac Gym 环境及缓存。官方 DiP、PHC、PPO 损失、奖励和任务采样不改；新的外部包装仅负责串行动作加载、实际仿真状态的单向遥测及正常退出时的显式资源清理。启用原生崩溃栈，不通过强制进程退出把异常包装成成功。已在独立核验中通过 GPU 内核和真实配置组合检查。

恢复队列 `r3_closd_reproduction_20261001` 依次执行：3072 环境三次更新的容量/退出诊断（不用作论文成绩）；官方已发布微调/未微调检查点各以4096环境、500帧、CFG7.5评估；实际场景本地可视 Gate 和容量通过后，按官方3072环境、CFG2.5从未微调 epoch62000继续4000轮；最终按4096环境正式评估。原始资源与新权重不上传 GitHub。

首次外部包装在任何 PPO 更新前因 Hydra 配置目录搜索错误退出1，已保留 `r3_closd_capacity3072_20261001_hydra_failed_v1.log` 并修复为绝对目录组合配置。第二次于14:21 UTC启动，14:29 UTC仍在3072环境初始化/串行动作加载，尚无 PPO epoch 输出；不能称容量检查通过或正式4000轮训练已经开始。GPU0用于CLoSD、GPU1用于DreamControl及独立渲染；以实际显存和退出结果验收，不以进程存在验收。

独立画面接收器 `scripts/r3_closd_scene_receiver.py` 经本机非阻塞UDP接收真实Isaac Gym刚体与物体状态，将角色和目标物体送到MeshCat场景服务。本地入口仍为 http://127.0.0.1:7001/static/。这是实际仿真状态抽样的浏览器渲染，不是Isaac Gym原生Viewer、WebRTC或检查点回放；初始化尚未发包时不能把旧画面误称当前训练。正式微调需记录该通道当前实际场景的可视验收。

运行协议与正常退出验收见 `docs/closd_reproduction_20261001.md`。官方评估器打印每状态至少1000次重置后的表格并主动以0退出，包装只接受此种评估退出并执行清理；非零退出仍判失败。踢击和出拳继续按上游共用STRIKE计数报告。

14:38 UTC 更新：3072环境短程容量/生命周期诊断已正常完成，state为 `COMPLETE stage=capacity3072 exit=0`。CPU读取检查点确认保存epoch62003，SHA-256为 `a4b86c32a1e2df3441bf28bc96a1c4686b79f2670138a5258b18a38e0c387b94`；日志依次记录官方runner正常返回、原生清理开始和完成。结构化证据在 `results/r3_closd_capacity3072_20261001.json`。这验证三次更新与正常退出，没有验证4000轮微调效果或旧139的具体根因。队列已自动转入官方已发布微调检查点的4096环境评估，随后评估未微调检查点；新的正式微调仍须当前实际场景可视验收。

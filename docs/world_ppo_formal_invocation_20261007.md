# 正式残差世界模型+PPO：唯一匹配预算队列

新fresh doc-follow代理仅读run-experiment skill、docs/dreamcontrol_ssh_environment_ledger.md与本文。用户已授权继续世界模型+PPO创新可行性验证，并明确确认当前6013/live真实Isaac Lab姿态持续运动。只有runtime、capacity、eval-interface三独立审计已Ready receipts、预测三seed门、场景Gate全部存在并通过才可调用；缺任一则保全停止，不代父代理写Ready或新Gate。

本地D:\具身智能\conditional-human-motion-generation执行唯一启动器一次：

```
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe scripts/launch_world_ppo_formal_20261007.py
```

GPU0 RTX A6000串行、GPU1既有独立renderer/6013隧道不改；进程间UDP40123非阻塞，页面断开不停止PPO。不用旧checkpoint回放替代实际状态。无环境依赖变化，spec configs/env-world-ppo-formal-20261007.json，source configs/world_ppo_formal_sources_20261007.json。未知SSH结果先只读tmux/state/output，不调用第二次；stage输出存在绝不重跑/覆盖。

tmux r3_world_ppo_formal_20261007，脚本r3_world_ppo_formal_queue_20261007.py。state artifacts/monitoring/r3_world_ppo_formal_20261007.state、调度日志同名.queue.log。顺序seed42、43、44，每seed P0→P-cap→P-history→W-correct→W-shuffled-original，各原66Kick参考、8192环境、24步rollout、500追加PPO更新；每臂98,304,000真实转移，合计1,474,560,000。原模型2000轮warmstart，恢复完整17参数Adam/LR，不冒充从零2000轮严格论文复现。

每训练后仅冻结model_update0500.pt在固定新11参考评2048环境500帧、0PPO/aux/norm；每臂1,024,000评价转移，全15臂15,360,000。细节在plans/world_residual_ppo_control_metrics_20261007.md；训练和评估每stage立刻捕获实际childrc、哨兵与COMPLETE，失败保全，不把无分数故障记0、不性能早停对照。训练每stage86400秒+60秒、eval1800秒+30秒，均服务器后台不阻塞对话。根据两8192容量诊断仅作主体更新吞吐外推，每seed/arm约3.6–4小时，15臂约2.3–2.6天，实际时长可能变化；不是承诺完成时间。

正式冻结评价使用v3精确文件名映射，将固定manifest编号映射到未重排的MotionLib目录加载索引；原先v1隐藏异常与v2顺序断言失败记录保留。v3两臂64×120接口均实际child0，33项运行/身份检查和19项独立trace重算通过，逐参考/宏统计误差0。回执为PASS_WITH_EVIDENCE_LIMITS：不宣称完整力张量、逐帧参数冻结、成功事件增量以及未出现的timeout/root失败已被独立实证；Ready只覆盖可启动的评估接口，非控制效果Gate。见results/world_residual_ppo_20261007/raw/eval_interface_v3/eval_doc_witness.json。

输出artifacts/world_residual_ppo_20261007/formal/{train,eval}_<arm>_seed<seed>_v1。训练v3在已容量验证v2上仅增加原wrapper clip_actions=None硬检查、读-only训练成功计数/参数计数和世界模型PPO边界文件；不改reward/PD/PPO/预算。必须核查初始原policy/logprob/value等价、实际WORLD_PPO_UPDATE逐次增加、progress实际转移、每50更新checkpoint与TensorBoard、数值有限、W不经PPO更新、交接KL<=.01、原66 identity/无额外IK。不是模型想象reward。

鲜代理启动后只读核验P0 seed42实际至少两个WORLD_PPO_UPDATE、progress、GPU进程、真实截图capture/step变化、独立viewer原PID保持，写launch回执results/world_residual_ppo_20261007/raw/formal_launch；不阻塞等所有15臂数日结束、不替人确认画面、不改代码、预算或stage。初始无迭代只能报初始化，不报训练推进；新v3入口如发生接口错误，保全精确stderr/rc等待父处理，不能自动修复或重跑。

只读状态命令（同一工作目录）：

```
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe scripts/read_world_ppo_formal_status_20261007.py
```

此命令不启动或修改任何训练；完整输出可保存在上述formal_launch回执目录。GPU0锁绝对路径为/data/code/ouyanghao/oyh/conditional-human-motion-generation/artifacts/monitoring/r3_world_ppo_gpu0.lock。首次文档调用前按skill只读nvidia-smi；实际共享viewer context允许<1500MiB，正式launcher还验证此锁空闲。SSH只读连接超时可最多两次重试，间隔至少10秒；唯一launcher结果未知时，先用状态命令查.dispatch.json、.state及tmux，绝不再次调用launcher。初始化等待期间观察至少30秒间隔，不要求反复读取整个日志。退出状态与实际进度必须一致；仅tmux存在不能认定已训练。

评分不得用于训练调参或挑模型。末尾五臂×三seed最终门PASS也不等于完整创新成立，必须后续两个同起点解释臂、Isaac因果通过后才MuJoCo。主门FAIL停止该候选，不增预算/阈值救结果。原生scene trace不是渲染视频，evaluation_video_verified=false直到独立渲染核验；保存给后续视频制作。许可轨迹/权重/原生数组留服务器，只metadata/日志/代码/图表进新私有仓库，旧格斗仓库保持。

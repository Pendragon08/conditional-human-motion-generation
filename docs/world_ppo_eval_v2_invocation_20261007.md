# 冻结控制评估接口 v2：保全首轮并先暴露原始异常

Fresh agent 仅读 run-experiment skill、docs/dreamcontrol_ssh_environment_ledger.md 和本文，再原样调用一次。用户已确认6013/live机器人持续运动；不代用户确认、不启动正式PPO。

v1在64环境初始化/首次reset后未进入有证据的评价循环，未写任何trace/summary/完成哨兵，连续卡在app.close。父代理先保全精确PID、命令、原日志/hash和空输出，随后仅SIGTERM失败诊断。实际child chain返回0，stage因哨兵缺失exit1、queue FAILED；不能当成功、自然超时或0分。科学源/权重不改，原目录保留。v2只增加初始化围栏打印、在清理之前输出原异常和对未进入主try的环境清理，未改变参考分配、推力、指标、策略或任何训练预算。最大300秒+30秒诊断上限防止同类清理继续吞噬时间；formal评估原1800秒上限不变。

新spec configs/env-world-ppo-eval-interface-v2-20261007.json、source lock configs/world_ppo_eval_sources_v2_20261007.json；依赖仍warm base47f5eb26，无安装。GPU0实际共享context <1500MiB，GPU1 renderer不动，lock=/data/code/ouyanghao/oyh/conditional-human-motion-generation/artifacts/monitoring/r3_world_ppo_gpu0.lock。启动前按skill只读核对GPU/lock与v1自然释放。SSH只读连接失败最多两次间隔10秒重试；启动结果未知先查tmux/state/output，绝不再次调用launcher。

本地D:\具身智能\conditional-human-motion-generation原样唯一调用：

```
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe scripts/launch_world_ppo_eval_interface_v2_20261007.py
```

唯一tmux r3_world_ppo_eval_interface_v2_20261007；state/log前缀artifacts/monitoring/r3_world_ppo_eval_interface_v2_20261007。每臂state/log前缀r3_world_ppo_eval-diagnostic_<arm>_42_20261007_v2，输出artifacts/world_residual_ppo_20261007/eval_interface_v2/eval-diagnostic_<arm>_seed42_v1。P-history再W-correct各64×120，冻结已完成capacity model_update0002.pt，0PPO/aux/norm。新11hash和manifest不变，phase0，外生reference/domain/sign表；80N torso推力step50..59，恢复窗口60..109。不是formal成绩，不以分数调方法。

失败时保存EVAL_ORIGINAL_EXCEPTION_BEFORE_CLEANUP原栈、实际rc/state、初始化最后阶段与空/非空输出。不能修代码、重试、安装或凭源码HASH宣布接口通过。成功时须独立读取服务器npz（CPU即可）重建64环境分配、first-alive/首done失效插补、两种failure原因与timeout、per-reference指标、10次force audit、冻结hash前后及0更新；scene trace不是视频。只回传metadata/日志，不下载许可数组或权重。

回执results/world_residual_ppo_20261007/raw/eval_interface_v2/eval_doc_witness.json，至少记录唯一launcher次数、actualchildrc/sentinel、8source SHA、11参考SHA和checkpoint SHA、独立trace checks/失败原因。任何接口FAIL均不得写formal Ready或启动训练。监控至少30秒间隔，等待与工具调用不超过60秒以保持沟通。

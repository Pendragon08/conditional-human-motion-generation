# 冻结控制评价接口 v3：以文件名核验固定参考分配

Fresh agent只读取run-experiment skill、docs/dreamcontrol_ssh_environment_ledger.md及本文。新源码/spec均未运行，必须独立按文只调用一次；无修复、重试、安装、viewer变更或formal训练。

前两次64环境预检都没有进入评价循环、没有任何trace/summary/完成哨兵，W阶段未启动。v1原异常被app.close清理遮蔽卡住，父代理保全后终止，childchain0但stage1失败。v2只增加异常输出/完整清理，自然结束，暴露Reference order must match fixed manifest：目录glob实际顺序为[0007,0009,0013,0012,0010,0006,0004,0000,0001,0015,0002]，与清单顺序不同；仍child0但stage1失败，不能写成功。两个失败目录/来源/日志保留，未选择模型、重采样或取得评分。

v3只以精确文件名将预注册manifest reference ID映射到未修改MotionLib库索引。首次reset每环境仍使用原manifest reference/domain/sign外生分配表，q/root/phase0规则不变；后续auto-reset仍官方函数。不按glob顺序改样本标签，不重排/改变MotionLib张量、模型或奖励；评估trace保存两种索引、文件列表及映射，summary每reference行用manifest文件名。保留v2原异常打印。依赖warm base47f5eb26不变。

GPU0 shared context<1500MiB、GPU1 renderer不动；lock绝对路径/data/code/ouyanghao/oyh/conditional-human-motion-generation/artifacts/monitoring/r3_world_ppo_gpu0.lock。先只读nvidia-smi/锁、两个旧tmux/process结束。只读SSH失败最多2次、至少10秒间隔；科学launcher未知返回先查state/tmux/out，不再次调用。

本地D:\具身智能\conditional-human-motion-generation唯一原样调用：

```
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe scripts/launch_world_ppo_eval_interface_v3_20261007.py
```

spec configs/env-world-ppo-eval-interface-v3-20261007.json，source configs/world_ppo_eval_sources_v3_20261007.json。tmux r3_world_ppo_eval_interface_v3_20261007；state/log前缀同名。单臂前缀r3_world_ppo_eval-diagnostic_<arm>_42_20261007_v3，输出artifacts/world_residual_ppo_20261007/eval_interface_v3/eval-diagnostic_<arm>_seed42_v1。

P-history→W-correct各64环境120帧，冻结capacity model_update0002.pt；0PPO/aux/norm，诊断墙钟300s+30s，不是正式分数，formal最终2048×500界限不改。相同新11manifest、80N torso世界Y±推力step50..59及恢复60..109。每stage须childrc0、完成哨兵、summary/trace齐全，sourcehash不能替代实际runtime。

成功后服务器CPU只读npz独立重建：manifest_to_library必须按精确filename匹配、initial_library_id逐env对应、外生reference/domain/sign表；first-alive与首done、后续2rad失效罚值，root/angle失败原因及time_out；每reference recovery/clean/success/failure/timeout/survival汇总，force audit10次；策略/norm/世界模型before-afterhash不变、0更新。若自然失败/timeout未出现，如实写此项仅结构校验，不假造实测事件；归一化Ridge文件/原模型身份核验。native scene数组不是视频，metadata回传、不下载权重或数组。

失败记录原异常打印、最后初始化阶段、actualrc与state、产物是否存在，不修复/重跑/scoregate。回执results/world_residual_ppo_20261007/raw/eval_interface_v3/eval_doc_witness.json，包括唯一调用次数、8source SHA、11reference SHA、checkpoint SHA、独立checks。全部通过才允许父代理标Ready；代理不能写Ready或launch formal。

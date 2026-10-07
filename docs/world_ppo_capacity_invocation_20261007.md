# 8192环境资源与系统不变量独立见证

仅读run-experiment skill、docs/dreamcontrol_ssh_environment_ledger.md与本文，不根据父代理先前训练判断成功；不修复、不安装、不重试、不启动formal。新版本仅加入capacity模式/证据快照，保留实际已执行v1及原五256环境smoke。固定环境warm reuse，无依赖变化。

GPU0串行、GPU1已有独立viewer保持；待原1200秒visibility自然exit0且旧见证CUDA只读审计结束后，检查GPU0<1500MiB、锁可得、tmux/state/output不存在。主执行者已记录用户明确确认6013/live运动的新Gate；这不是见证者代替用户判断。

本地repo执行一次：

```
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe scripts/launch_world_ppo_capacity_20261007.py
```

唯一tmux r3_world_ppo_capacity_20261007；state/log r3_world_ppo_capacity_20261007.*。先零PPO/零sim CUDA fixture，120秒时限；后P-history与W-correct各8192环境×24×2诊断PPO、seed42，W每次1×1024辅助更新，分别1800秒时限+30秒TERM→KILL。child rc紧接capture，state COMPLETE且0/sentinel三阶段缺一则FAILED，保全不改配方重跑。

输出artifacts/world_residual_ppo_20261007/invariant_fixture_v1与capacity_<arm>_seed42_v2。独立载入fixture_tensors.pt逐张量核验single pre-action timeout bootstrap，真终止无bootstrap，reset切断10帧历史，有序非reset历史保留。每capacity 393216真实转移、2diagnostic/0formalPPO、W2aux；8192完整rollout/storage及P-history5×4 minibatch反传、W auxiliary所需显存/墙钟核验。检查每次finite loss/grad、PPO不改W/norm、aux不改actor/Ridge、handoff KL<=.01、原66参考/原checkpoint identity。额外timeout_fixture.pt只是非物理系统fixture，不与自然reset混淆。

只回传summary/progress/state/log/hash及独立回执到results/world_residual_ppo_20261007/raw/capacity。权重/fixture张量留服务器；原checkpoint SHA e3c50fb6f36f59f89d1b4e788b1ba74cf1ff0cdde3dceff34a9d7765235f65c0，旧资产目录model_1999.pt与接口Kick_references.npz路径见runner常量及docs/world_residual_ppo_runtime_invocation_20261007.md。通过仅表示容量和系统语义，不是500轮控制效果或创新点验证。结束后不自动正式训练。

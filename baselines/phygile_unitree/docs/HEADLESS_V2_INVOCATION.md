# 本次授权的无界面续训见证

2026-10-22 用户明确允许本次无界面续训，原文保存于 `configs/headless_authorization_20261022.json`。此例外不修改旧 GUI 要求和冻结运行，也不修改 ReactiveBFM。

环境按 `docs/COMPUTE_LEDGER.md` 与 `configs/environment_plan_v2.json` 暖复用，没有新增依赖。新代码已部署到服务器 `code_v9`。启动器会检查 GPU1 至少余 6GiB、磁盘至少余 1GiB；使用 CPU16–19、四线程、GPU1，PyTorch 分配上限 20%。

独立见证者只执行下面这条本地 PowerShell 指令一次，不自行改代码或重试。输出目录必须不存在。

```powershell
& 'C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree/scripts/submit_isaac_job.py' --version code_v9 --name headless_smoke_v3 --mode train --num-envs 64 --iterations 2
```

返回的 PID 是本次独立任务。通过 SSH 只读检查 `/data/code/ouyanghao/oyh/phygile-unitree-20261022/runs/headless_smoke_v3/receipt.json` 及同根目录 `logs/headless_smoke_v3_stdout.txt`、`logs/headless_smoke_v3_stderr.txt`。轮询不要超过每30秒一次，单次等待不超过60秒；总等待上限10分钟。失败如实报告，不重跑，不触碰其他进程。

完成要求：`status=BOUNDED_TRAINING_COMPLETED`、`headless=true`、`optimizer_loaded_exactly=true`、`additional_ppo_iterations=2`、`actual_adam_calls=40`、`changed_policy_tensors>0`、`executed_sources_unchanged=true`、`parent_unchanged=true`。记录完整命令、输出、退出码/启动状态及实际回执，不把启动成功当成训练完成。回执应绑定 `executed_sources.json`，其 SHA256 必须匹配。

本次仅验证恢复和实际优化，不用于证明动作质量提高。保留全部记录；原始动作与模型不上传公开仓库，发布清单与指标。无飞书、邮件或定时任务。

V2 启动仅在参数解析阶段失败。V3 将自定义参数改名为 --execution-consent 并禁用 argparse 缩写，避免 AppLauncher 注册 --headless 前的预解析冲突；保留 V2 原始失败记录。

# 独立文档见证 V3：PASS

按 docs/HEADLESS_V2_INVOCATION.md 原样执行 PowerShell 提交一次，未修改代码、未重试。使用 run-experiment 技能及 compute-env-contract 的 fresh agent-follows-doc 约定。环境规范规范化 SHA256 为 a43f960bd3bcdcd3398655bffe291138b73de5ca7f30901765172ff9821e1954，与计算账本一致，暖复用原环境。

训练在 2026-10-22T11:58:00.544908+00:00 至 2026-10-22T11:58:49.456265+00:00 完成，耗时 48.911 秒。独立 PID 为 3157748；第一次只读轮询时进程已退出。轮询 UTC 为 2026-10-22T11:59:08.304289+00:00，与提交间隔超过 30 秒；没有超过 60 秒的等待，没有超过 10 分钟总等待。提交器退出码为 0。分离子进程的原始退出码未由启动器保存，本报告不将其假定为 0。

## 实际完成证据

| 验收项 | 实测 |
|---|---|
| status | BOUNDED_TRAINING_COMPLETED |
| headless / DISPLAY | true / null |
| 环境数 / 新增 PPO 迭代 | 64 / 2 |
| 优化器、策略、归一化器精确恢复 | true / true / true |
| 实际 Adam 调用 | 40 |
| Adam step | 40000 → 40040 |
| 变化的独立 named_parameters 张量 | 21 |
| 变化的 model_state_dict 条目 | 27（包含共享模块别名，统计口径不同） |
| 最终策略/模型张量有限性 | 97 个张量全部有限 |
| 最终优化器张量有限性 | 75 个张量全部有限 |
| 控制调用 | 48 |
| 已执行源文件 | 446 个；独立退出后检查全部存在且哈希未变 |
| 启动前代码文件 | 336 个；独立复核全部未变 |
| 父检查点 | 启动前、回执、结束后 SHA256 一致 |
| GPU1 峰值 Torch 分配 | 580916224 字节 |

源代码的递归逐项比较在设置 optimizer_loaded_exactly=true 之前检查恢复优化器与父检查点完全相等；每次 Adam 更新前检查梯度有限，训练结束检查策略和优化器状态有限。本次另用原环境只读加载父/最终检查点、验证张量有限性及变化，未启动新的模拟或训练。

## 绑定哈希

- executed_sources.json：ecc1a7d93cdc310c6268f11c00b701b7a28598af59371b59cf28d4203f36540b；与回执 source_provenance.sha256 完全一致。
- 执行入口 lab_run.py：8afffc1fade430861dab8c96b38ba6ca7d6c2eef2aca2878c72180a50ed6c0f2；与实际源文件和 executed_sources.json 一致。
- 父检查点：bb067f364de74e2da7cf1879e6f26bb961a71eb9ff64cbfbe84dd858bfae2336。
- 新检查点：cf2d6fac12394764df9fc694c14dc87eece2bbdfc35b8c63cca78e921ff548fe；独立只读复核一致。

完整启动命令在 documented_submission.ps1；原始提交结果在 submission_result.json。实际服务器启动记录、receipt.json、executed_sources.json、contract.json、stdout/stderr 和部署参数已按原始字节保存于 remote_artifacts/，每个副本的 SHA256 与服务器读取值一致。完整哈希见 artifact_hashes.json；独立审计见 independent_audit.json。检查点继续保存在服务器原位置，本次不复制或改写其内容。

## 文档与实际差异

本次完整调用未发现参数、资源选择或完成语义方面的文档差异。原环境配置保留旧 GUI 条件，文档及执行授权记录明确允许本次 headless 例外，因此未请求 GUI 确认。

stderr 有一行 DriverShaderCacheManager::init() called without a shutdown()!，已原样保留。该诊断未阻止实际 40 次优化、有限参数更新和最终回执产生；本报告没有删除或掩盖它。

本见证只验证恢复及真实的有界优化，不用于证明动作质量、跨仿真稳定性或论文基准已提高。没有操作 ReactiveBFM、原始数据集或其他任务；没有修改源文件、修补、重试、通知或创建定时任务。

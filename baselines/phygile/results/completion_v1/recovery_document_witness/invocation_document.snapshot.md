# 独立站立边界与确定性修复见证

先前 `results/completion_v1/closure_final` 已失败，原样保留。本次使用新目录，不覆盖、不放宽阈值。新增确定性设置：CUBLAS workspace、PyTorch deterministic algorithms、禁用 TF32、PhysX enhanced determinism；保存每个子进程的初始物理参数。它是工程配置，不声称作者默认配置。

SSH 服务器：`ouyanghao@10.147.17.32`，端口 22；本机密钥 `C:/Users/Pende/.ssh/id_ed25519`。只用 GPU0；已有闲置占用获用户授权复用，不停止其他进程。先检查服务器无本任务运行，GPU0 约 3 GiB 闲置占用。

独立见证者在远端目录 `/data/code/ouyanghao/oyh/phygile-reproduction-20261017` 执行以下命令一次：

```bash
bash scripts/run_isaac.sh scripts/validate_terminal_boundary.py --name closure_recovery
```

脚本 600 秒上限，若运行超过 60 秒，保持读日志并向主任务反馈。禁止自行修代码、改变阈值或重跑。SSH 可用 Python subprocess 参数列表传递命令，避免 PowerShell 管道额外 CR 空行。无需默认系统 Python，它没有 NumPy。

期望 `results/completion_v1/closure_recovery/verification.json` 和 `output/verification.json` 存在，外层退出 0，`PHYGILE_TERMINAL_BOUNDARY_COMPLETE` 恰好一次。核对独立重放原始状态/物理参数差异；50 帧候选末帧与独立的零速度名义 G1 站立目标完全相同。站立是目标约束，不是实测稳定站立。随机生成器可以拒绝全部候选，不能冒充成功生成。两个各 1 帧保持姿态窗口仅为接受—反馈接口代理见证，不是正式性能指标。

本次无训练后的生成器，无闭环 PPO 更新；配对数据仍缺。主 2000 轮模型冻结。保存文档 SHA、原命令、stdout/stderr、退出码、结果/NPZ 哈希与报告到 `results/completion_v1/recovery_document_witness/`。失败须保留，报告真实失败。

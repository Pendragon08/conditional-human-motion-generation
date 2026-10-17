# 名义站立末端补充验证

复用 `docs/environment.md` 中的 spec `fa5883c8`、原服务器和授权 GPU 0 共享条件，不改安装。此验证在完整的 `document_run` 已完成后执行一次，保留此前所有证据。

SSH 使用既有身份 `ouyanghao@10.147.17.32`、默认端口 22、已有密钥 `C:/Users/Pende/.ssh/id_ed25519`；使用 BatchMode=yes 和 ConnectTimeout=15，不修改 SSH 配置。

```sh
cd /data/code/ouyanghao/oyh/phygile-reproduction-20261017
bash scripts/run_isaac.sh scripts/validate_terminal_boundary.py
```

输出目录 `results/completion_v1/closure_final/` 必须尚不存在；存在时不改名、不覆盖、不重跑。最长 600 秒。必须自然退出 0，包含 `PHYGILE_TERMINAL_BOUNDARY_COMPLETE`，且 `output/verification.json` 的 status 为 `COMPLETED_ENGINEERING_VALIDATION`。

新末端目标从固定 G1 配置的名义站姿生成，含 hip/knee/ankle 与上肢角度、0.76 m 根高度、零速度；坐标平移/朝向与初始实际状态对齐。它是明确的名义站立目标，不是作者公开后缀，也未单独证明动态站立稳定性。原先两个各 1 帧的保持初始化姿态测试仍明确单列，不能当作站立或语义生成成功。

这里再次执行真实随机权重扩散候选、262D 解码、1 秒跟踪、接受/拒绝/重放，只检查改为独立站立目标后的接口。候选可以全部被拒绝；不调整容差、不宣称生成器训练或闭环 PPO 适配已完成。记录原始候选、末端目标、实际状态、回滚请求、代码哈希和所有结果。

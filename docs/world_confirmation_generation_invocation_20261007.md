# 新确认动作生成：文档执行见证

先读run-experiment及compute契约、configs/env-world-confirmation-generation-20261007.json、plans/world_residual_confirmation_20261007.md。服务器ouyanghao@10.147.17.32，源10.147.17.250，私钥C:/Users/Pende/.ssh/id_ed25519，ROOT=/data/code/ouyanghao/oyh/conditional-human-motion-generation。原环境只读复用，不装包；GPU0开发队列已COMPLETE，约699MiB空闲基底，GPU1旧独立viewer不动，/data约52GiB可用。

唯一确切调用一次：

```
tmux new-session -d -s r3_world_confirmation_generation_20261007 'cd /data/code/ouyanghao/oyh/conditional-human-motion-generation; bash scripts/r3_world_confirmation_generation_20261007.sh > artifacts/monitoring/r3_world_confirmation_generation_20261007.log 2>&1'
```

不要重试未知启动结果，先只读查state/session。不要改代码、安装、换seed/数量/阈值、再次抽样。GPU0锁和已有输出防重复。各阶段timeout分别1800/3600/900/600s；失败保全所有产物和日志，不宣称环境ready。

预期两个既有runtime各真实seeded CUDA backward，retarget环境另做JAX GPU JIT witness，有限且sentinel WORLD_CONFIRMATION_RUNTIME_WITNESS。随后真实OmniControl DDPM32条196帧输出；逐字原官方PyRoki retarget与refine后，215帧G1参考，仅按预锁静态标准首11数字ID复制到selection/selected_refs。需核对权重sha、源码原样hash、实际生成shape/有限、接受/拒绝总分母32、11选中hash及manifest。不足11为合法BLOCKED，不重试。所有动力学和预测未运行，formal_PPO=false；selected refs必须生成后与原66及候选去重（消除初始root平移/yaw且不时间warping）且未按执行成绩挑选。

阶段state/log前缀artifacts/monitoring/r3_world_confirmation_generation_20261007；manifest artifacts/world_confirmation_20261007/selection/manifest.json。Stage原源码在third_party/DreamControl/TrajGen/sample_world_confirm_20261007，rendered diffusion videos不是物理训练视频。最终exit0、state COMPLETE和WORLD_CONFIRMATION_GENERATION_COMPLETE/WORLD_FRESH_CONFIRMATION_REFERENCES_COMPLETE才算执行完成，不能算预测门或控制门通过。

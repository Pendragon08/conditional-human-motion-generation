# 残差世界模型与真实PPO：独立运行见证

适用run-experiment的fresh agent-follows-doc检查。仅读本文、该skill及docs/dreamcontrol_ssh_environment_ledger.md，不凭先前运行确认接口。使用现有环境；不安装依赖，不改代码、规则或执行预算。

## 部署及权限边界

本地D:\具身智能\conditional-human-motion-generation；服务器ouyanghao@10.147.17.32，/data/code/ouyanghao/oyh/conditional-human-motion-generation。SSH -b 10.147.17.250、私钥C:/Users/Pende/.ssh/id_ed25519。新源码由主执行者同步，configs/world_residual_ppo_runtime_sources_20261007.json列出确切哈希。固定Isaac Sim4.5/Python3.10/Torch2.5.1cu118、原DreamControl Isaac Lab任务，无依赖变化。

唯一诊断tmux r3_world_residual_ppo_smoke_20261007。GPU0串行算力，GPU1原viewer保持。GPU0允许既有viewer跨卡约699MiB上下文，启动器要求<1500MiB、计算锁可得、磁盘>15GiB；如有别的compute进程须停止启动检查，不中断它。确认v2三seedclean/recovery预测PASS后才可做诊断；不是正式训练。

在本地repo执行一次：

```
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe scripts/launch_world_residual_ppo_smoke_20261007.py
```

未知SSH返回结果先只读查tmux/state/output，不盲重试。有诊断输出或state则不得再调用启动器。state前缀artifacts/monitoring/r3_world_residual_ppo_smoke_20261007。依次P0、P-cap、P-history、W-correct、W-shuffled-original，各256环境×24步×2真实PPO更新，seed42；每个W两次辅助更新(每周期1步×1024batch)，P对照无aux。各arm新artifacts/world_residual_ppo_20261007/smoke_<arm>_seed42_v1输出。各脚本1200s时限TERM+30sKILL；立刻捕获child rc、哨兵WORLD_RESIDUAL_PPO_CONTROL_COMPLETE，均须0和完整summary。程序失败不是控制方法成绩，不替换源或重跑失败arm；保全state/log。

全部五臂smoke通过后，队列自动做16环境冻结原Kick策略的1200秒实际物理可视预检。脚本r3_world_residual_visibility_20261007.sh，时限1500s+30s，实际IsaacUDP40123→GPU1独立renderer→6013网页；无PPO。阶段只读画面label必须为画面预检(仅推理)，不是正在正式训练。结束后STALE只代表预检结束。不修改viewer、不代替人类确认，不生成正式Gate，不启动formal队列。

## 必须独立核验

1. 五summary及各child_exit=0，formal_PPO_updates=0、diagnostic=2，原66参考identity<1e-6，初始actor mean/value/std/logprob误差在1e-5/1e-5/0/1e-4内。
2. 原17(以实际为准)命名Adam参数moments/step逐张量等同；新增首层128列moment为0，新GRUstate为空；原LR同步恢复至alg和optimizer。unused critic norm count0没有用，critic与actor共享冻结原norm。
3. History有序10帧、reset前历史切断。P-history/cap初始GRU一致、cap不看前动作、W只读h，不读取未来；PPO epoch0 ratio误差<1e-4；PPO不改W或norm。
4. isolated timeout fixture真终止/截断区分；每实际转移只有一次官方pre-action-value bootstrap，误差<1e-6，不声称精确terminal-state bootstrap。
5. W在线辅助窗口t9..14，完整history10/future10/reset-free；只用原66真实rollout。错误监督donor在clean/phase/contact/sign层内确切双射、不同reference/episode，无历史未来重叠；delta=当前Ridge+donor残差；contacts单独donor。共同不可匹配行仅aux排除，控制分母保留。
6. PPO后辅助可改变W，但actor/critic/Ridge/各norm不变；版本交接KL<=.01，不按回报自适应重新跑。correct/shuffle从selected weights加载，auxAdam统一fresh；不加载末1000step错时optimizer。
7. 两真实PPO更新policy_changed、loss/梯度有限、checkpoint/tensorboard、实际转移/aux预算/墙钟/显存记录完整。接口smoke不验证控制收益，也不证明8192容量或500更新完成。
8. 原预测确认/模型/数组和其他GPU进程哈希/身份保持；无依赖安装/额外IK/imagined奖励。仅metadata回传本地results/world_residual_ppo_20261007/raw，许可权重/数组不上传。

写fresh-doc回执，记录实际错误/差异而非修复它。核验输出后主执行者才能准备正式runner；新formal必须当前实际场景本地用户确认且进一步8192容量见证，不能用旧直播/检查点回放代替。

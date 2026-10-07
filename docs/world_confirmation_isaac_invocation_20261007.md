# 新动作物理预测确认：文档见证

环境warm reuse基础configs/env-dreamcontrol-r3.json，规范SHA47f5eb26218254bbe270c3c43b3e7a552cca04c5caa81b6c76b4a6d2d3d2e9d4，无安装。读取docs/dreamcontrol_ssh_environment_ledger.md、plans/world_residual_confirmation_20261007.md与run-experiment/compute契约。SSH ouyanghao@10.147.17.32、源10.147.17.250、私钥C:/Users/Pende/.ssh/id_ed25519。ROOT=/data/code/ouyanghao/oyh/conditional-human-motion-generation；GPU0计算，GPU1独立viewer不动。

只在新生成静态manifest COMPLETE选中11条、生成队列正常结束后执行一次：

2026-10-07首个文档见证经PowerShell管道把CR带入远端tmux命令，空log、无session/state/model_lock/采集证明科学队列没有开始。保全原log时间及独立诊断，不当方法FAIL。修订只改变运输；科学queue/stage/eval源码、协议、模型和参考hash不变。

在本地仓库工作目录严格执行以下命令一次；它经Python参数列表/base64发送固定LF程序，避免PowerShell管道换行，并在dispatch前断言没有真实队列/输出。不会生成新数据或修改科学脚本。返回成功仍需只读核实tmux/state实际存在。

```
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe scripts/launch_world_confirmation_transport_20261007.py
```

内部tmux参数为原r3_world_confirmation_20261007、原queue，stdout改独立r3_world_confirmation_20261007_transport_v2.log；原空logmtime保全到r3_world_confirmation_20261007_crlf_dispatch_receipt.json。禁止再次用here-string经SSH bash -s发送此启动命令。

不修改/修复/安装/再次选择；未知启动先只读查session/state。输出已存在拒绝，GPU0锁、磁盘低于15GiB阻塞；每collect上限1800s、evaluate900s、TERM+30sKILL。启动前model_lock包含六模型、Ridge系数、normalizer/checkpoint、source/schema/decoder、参考manifest和全部选中参考hash，必须在采集前锁定。

按seed42/43/44 collect2048×480=983040真实转移，再evaluate全部六冻结模型，零PPO和零辅助更新。原27跟踪关节/29动作、626obs、原完整随机actor/std、原reward/PD/termination，仅参考换11新生成；原运动库截取前200帧并原样重算Kick事件。collector检查实际加载文件及hash、原joints顺序误差<1e-6、原JIT等价<1e-5。均为仅推理，不是正式RL。每seed1024clean/1024push，step50..59 ±80N实际forcebuffer10条；完整reset/episode/source记录。

确认每reference/域最多1024合法窗且至少256，没有把难动作或reset事件从11动作分母删除。窗口原始历史/actions/targets能逐行重建且不跨reset/reference，恢复60..109属于push起始episode；norm/模型不更新，retired原test不在新11参考。六模型共享完整窗，无donor删样，错误模型沿用开发后被冻结checkpoint。future-action-zero两路及update0还原ridge数值验证；未来实际动作不进入当前actor。主FK及CI见锁定计划。

结果artifacts/world_confirmation_20261007/{collect,evaluate}_seed{42,43,44}_v1/summary.json和confirmation_windows.npz；state/log前缀r3_world_confirmation_<mode>_<seed>_v1；总state/log r3_world_confirmation_20261007.*。六stage正常exit0及sentinel，最后WORLD_RESIDUAL_CONFIRMATION_QUEUE_COMPLETE PASS/FAIL均合法；程序错误/覆盖不足不能冒充0成功率或方法FAIL。确认FAIL不重复；PASS只允许之后准备PPO，队列不会启动正式训练。

检查每seed真实983040、总2,949,120；窗口、hash、norm和model_lock在前后不变、每状态独立指标、实际子进程退出码。wrapper在timeout返回后立即保存rc，不以哨兵替代正常退出。当前预测门/控制门分别报告，不把采集显示当正式PPO。数据/权重留服务器，许可资产不上传。

主统计文字已于第二运输见证未dispatch时更正，保持开发/确认实际代码sqrt(mean(windowMSE))→reference等权平均。原文本保全；本次以更正后锁定计划的hash采集，无任何结果后选指标。

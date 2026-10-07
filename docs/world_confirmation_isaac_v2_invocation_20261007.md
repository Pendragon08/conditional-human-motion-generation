# 新参考身份校验修正：确认v2独立见证

按run-experiment与compute契约，warm reuse configs/env-dreamcontrol-r3.json和docs/dreamcontrol_ssh_environment_ledger.md，无安装。读取plans/world_residual_confirmation_20261007.md；全部模型/normalizer/11参考/seed/动作任务/预算/门保持冻结。独立GPU1viewer不动，GPU0仅推理。

首运输CRLF失败科学执行0。后来唯一v1科学queue在实际参考身份guard报ValueError，lib.dof_pos为(11,200,27)而expected_q错误concat为(2200,27)，尚未入采集循环；无steps/npy/窗口/评估/43或44。错误后app.close卡住超过200秒，父agent只对精确失败PID1612527发一次SIGTERM清理；python.sh/timeout链实际返回0但无完成sentinel，stage/queue正确FAILED。不能称正常结束或方法FAIL。原39文件锁/source/state/log保留，独立transport_v3见证核验队列/子进程结束、GPU0锁释放。失败不代表0成功率。

v2独立文件r3_world_confirmation_isaac_v2_20261007.py只将身份预期concat换成stack，并assert实际(11,200,27)，不改变运动库、采集、模型、物理、随机化、窗口或指标。请独立比对与原collector其余源码完全一致。原已执行文件不改。v2queue/stage用新confirmation_v2目录与新r3_world_confirmation_v2_20261007前缀，manifest仍原selection。新模型锁必须与原模型/参考hash完全一致；source变化仅新collector身份guard和v2queue封装，plan更正后hash不变。

先只读确认无v2 session/state/art目录、原v1 FAILED且数据为空、旧PID全部消失、GPU0锁空闲、磁盘>15GiB。仅从本地仓库执行以下精确命令一次：

```
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe scripts/launch_world_confirmation_v2_20261007.py
```

helper使用base64/subprocess参数传输，不通过PowerShell here-string到SSH bash-s。未知dispatch先只读检查；绝不重复。精确队列r3_world_confirmation_v2_queue_20261007.py，tmux r3_world_confirmation_v2_20261007，stdout artifacts/monitoring/r3_world_confirmation_v2_20261007.log。

每seed42/43/44 collect2048×480=983040，再同数据6冻结模型eval一次。总2,949,120真实转移，0PPO/0辅助更新。原app.close改动为0；成功采集仍先env.close再app.close。新wrapper立即记录实际rc，必须rc0及sentinel，不能用SIGTERM后的0替代完成。程序错误/覆盖不足保全，不自动修复重跑。

输出artifacts/world_confirmation_20261007/confirmation_v2/{collect,evaluate}_seed{42,43,44}_v1/summary.json；stage state/log r3_world_confirmation_v2_<mode>_<seed>_v1，模型锁在confirmation_v2/model_lock.json。比对39模型/参考/原decoder/schema及计划hash；新collector与原唯一guard差异。窗口10历史/10未来同episode/reference，recovery60..109属于push初始episode；每ref每域≥256，完整done/reset/forcebuffer记录。主统计sqrt(mean(windowMSE))逐reference→11参考等权平均，CI/阈值/六模型锁定规则同原plan。

全部stage正常childrc0+sentinel后才能COMPLETE（预测门PASS或FAIL均有效科学结果），独立审核hash/窗口/归一化冻结/实际reset。PASS不是PPO收益；FAIL不得改模型、阈值或参考重test。许可数组权重留服务器，本地只归档metadata。

# 残差世界模型阶段：扰动数据接口见证

适用环境沿用 `configs/env-dreamcontrol-r3.json`，canonical SHA256 `47f5eb26218254bbe270c3c43b3e7a552cca04c5caa81b6c76b4a6d2d3d2e9d4`，不安装任何依赖。服务器10.147.17.32，ouyanghao，源10.147.17.250，私钥C:/Users/Pende/.ssh/id_ed25519。GPU0 A6000目前699MiB旧预览分配，保留；GPU1独立renderer。61GiB磁盘剩余；正式三种子采集预计增加7GiB。

Fresh witness只读该文档、运行skill及provider ledger `.aris/compute/ssh-isaac.md`。原样执行以下命令一次，不安装、改源码、修复或重试：

```
bash /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/r3_world_ppo_residual_stage_20261007.sh probe 42 docwitness_v1
```

该诊断16环境×480控制步=7680真实Isaac转移；加载原Kick完整checkpoint、原随机actor与原normalizer，奖励、PD、termination、66参考不改。外力为新增且单独记录的分布：偶数env干净，奇数env在step50..59受到世界y方向±80N、0.2秒torso力。外力不进入actor或encoder。记录推力开始episode、cohort、方向、所有原数据及10次Articulation外力buffer核对。它是扰动数据采集诊断，PPO更新必须为0，不能叫正式RL。

预期wrapper退出0，state `COMPLETE mode=probe exit=0 formal=false`，日志出现 `WORLD_PPO_ISAAC_COMPLETE`，stdout出现 `WORLD_RESIDUAL_DOCUMENTED_STAGE_COMPLETE mode=probe`。路径 `artifacts/world_residual_20261007/probe_seed42_docwitness_v1/summary.json`；state/log `artifacts/monitoring/r3_world_residual_probe_42_docwitness_v1.*`。900秒TERM+30秒KILL，不无限等待。

核对7680真实步数、0 PPO、旧JIT初始error<1e-5、原参考order error<1e-6、外力buffer10条且每条8个非零env、norm约80N、误差<1e-6。外力buffer不是接触或控制收益证明。11个.npy大小与hash须匹配summary；数据含有限状态、动作、reward，done/episode变动相符。原termination保留；force前后reset单独记录，不能排除失败环境后声称控制稳定。

该接口的通过不覆盖上一轮预测FAIL，也不批准正式PPO；完整预测对照与本轮本地可视门另立。结束后实际遥测停止，STALE不代表训练停滞。

# 世界模型与真实rollout PPO：运行见证 v3

用户2026-10-06授权验证，正式消融仍须预测门和新的本地实际Kick画面门。诊断不冒充正式结果。根目录服务器 `/data/code/ouyanghao/oyh/conditional-human-motion-generation`，SSH用户ouyanghao，主机10.147.17.32，源10.147.17.250，私钥C:/Users/Pende/.ssh/id_ed25519。

环境复用 `configs/env-dreamcontrol-r3.json`，规范JSON哈希47f5eb26218254bbe270c3c43b3e7a552cca04c5caa81b6c76b4a6d2d3d2e9d4。Isaac Sim4.5 Python3.10，Torch2.5.1+cu118。无新增依赖。GPU0 A6000，GPU1独立渲染；GPU0少量既有预览分配不属于正式训练，保留不动。服务器/data当前71GiB可用，三个warm-up约7GiB；每臂滚动保存训练监督窗口而非无界增长。不复制许可权重到GitHub。

## Fresh文档见证，唯一一次

仅执行以下确切调用。不要安装、改代码、修复、自动重试、切换设备，失败原样报告。作者不能自证。

```
bash /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/r3_world_ppo_stage_20261006.sh smoke 42 docwitness_v3
```

该调用16环境真实Isaac物理48步、两次24步rollout和原RSL PPO更新（每次5epoch/4minibatch）、一次真正CUDA世界模型辅助更新。奖励/PD/termination不变，完整原626观测与29动作，原checkpoint复制；确认初始actor与原JIT误差<1e-5，新latent权重为零，世界模型在PPOepoch中逐张量不变，真实参数两者都改变，预测窗口不跨reset。只做诊断不满足正式500轮门。预期日志 `WORLD_PPO_ISAAC_COMPLETE`、`WORLD_PPO_DOCUMENTED_STAGE_COMPLETE mode=smoke`，退出0；summary.json须真实更新为true、loss有限。

输出：`artifacts/world_ppo_20261006/smoke_seed42_docwitness_v3/`。状态/日志：`artifacts/monitoring/r3_world_ppo_smoke_42_docwitness_v3.state/.log`。共享GPU0文件锁防重复并发；已有输出即拒绝。stage脚本自身900秒TERM加30秒KILL的诊断时限，不无限等待。

v1未写任何物理step并达到有界超时，原日志/空输出保留。纯Torch数值诊断已经观察宽754矩阵零列相对626原网络有7.152557e-7差异，v1要求差异严格0不合理；这还不能证明v1停顿的全部原因。v2显式1e-5函数等价容差、异常在Kit关闭前打印，并每180秒导出栈。原policy参数、数据、奖励、物理不改。不能因import或纯Torch检查宣称Isaac PPO成功。

v2新鲜见证真实物理及PPO/aux更新成功，但进一步原runner审计发现此Kick没有privileged critic观测；原checkpoint的privileged normalizer count0是未用资产。v2诊断曾错误地用它处理critic。v3严格复用原runner的actor-normalized observation给critic，原baseline价值函数才能一致。warm-up只用actor且无PPO，因此三seed共享数据与预测门不受影响，旧smoke不覆盖或包装成正确正式训练。

v3还需summary五臂实际观测下初始action mean/value均<1e-5、log-prob<1e-4（float32累积容差）、历史reset补零独立fixture、官方timeout只加一次gamma*V的独立fixture均通过；这些fixture不是改真实仿真termination。真实rollout若无done不能说真实reset已经覆盖；warm-up的9/12/4真实reset另由窗口审计覆盖。

## 预注册数据/预测门

smoke通过之后，每seed42/43/44独立执行collect v1（8192环境480控制步＝3,932,160转移，无PPO更新），共享原冻结策略分布包括原std，不改物理扰动或参考。actual场景通过非阻塞UDP40123送独立viewer，独立监控失败不影响采集/训练。完整reference分组同一锁定划分，训练/validation/test标签不能交叉；actor原先见过66参考，因此不是策略未见动作泛化。

原观测前92维为可部署本体状态：3线速度、3角速度、3重力、27相对关节角、27速度、29上一动作。真实传感器状态仅监督：27关节、姿态/根速度、足位置、滑移、根高度、地面contact。10帧因果历史、128GRU、非仿射LayerNorm；H1/5/10未来动作只进预测头，不能进入当前actor。buffer跨reset断开。接触阈值5N是地面传感标签，不是实体打击成功。

P-cap复用相同GRU，每决策当前本体复制10次，previous-action通道置零，隐藏状态从零；只增加容量而不引入历史。P-history真10帧历史且PPO训练GRU。W latent止梯度，aux与PPO优化器参数分离，所有epoch用缓存同版本latent。所有臂原normalizer冻结为同checkpoint，encoder norm仅warm-up训练references拟合并冻结，history controls不载入正确WM encoder权重。

参数需分列部署、PPO、aux、总训练参数。P-cap/history严格匹配W部署参数；预测头只存在W训练期，不能说所有总训练参数相同。W-correct/shuffled严格匹配全部参数、梯度次数和样本。相同真实RL步数不等于同GPU总计算。该修正来自独立消融审查，五臂500更新8192环境三seed预算不变。

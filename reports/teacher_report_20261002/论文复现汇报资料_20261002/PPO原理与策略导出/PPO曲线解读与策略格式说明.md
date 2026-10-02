# 六任务PPO曲线解读、更新原理与策略文件

本次只读取已完成的六项基线logger、保存配置、部署接口和实际训练库源码，补充本地汇报HTML；未修改训练队列、模型、参考或评估结果。

## 曲线怎样支持结论

优先将训练reward与回合长度、奖励分项共同阅读，再以冻结策略官方评估、接触、MuJoCo结果判断实际效果。`Loss/value_function`是Critic预测误差而不是原始状态价值；`Loss/surrogate`是负裁剪代理目标而不是完整PPO损失。两者下降不能单独证明动作质量。

六项均有2000个迭代记录。前/末100轮算术均值、最高100轮窗口与来源CSV哈希在`results/ppo_training_interpretation_20261002/six_task_curve_analysis.json`。Punch滑动窗口不跨恢复点1600；窗口筛选仅描述已有reward，不用于重新挑选策略。单次训练窗口之间未做显著性检验。

| 任务 | reward前100→末100 | 曲线结论 |
|---|---:|---|
| Kick | −10.124→6.013 | 末期接近平台，官方成功率85.3% |
| Punch | −10.262→5.335 | 恢复后短暂退化，末期仍低于恢复前1500—1599均值6.405；官方87.2% |
| Jump | −9.914→5.597 | 最高100轮6.206后回落；Isaac100%与MuJoCo约2.9秒摔倒应共同呈现 |
| Sit | −10.373→7.046 | 末期平台；Isaac100%仅官方目标判据，MuJoCo无持续座椅接触 |
| Pick | −10.026→9.624 | 较最高100轮9.911小幅回落；Isaac93.9%，MuJoCo未抬物 |
| Bimanual-Pick | −10.811→5.990 | 末期仍改善，不能称充分收敛；单参考Isaac99.9%，MuJoCo未抬物 |

KL、clip fraction、explained variance、实际梯度范数未作为单独tag出现在这六份导出logger中。实际学习率有记录；不能据损失曲线补造缺失诊断。`Episode_Reward/*`是加权奖励/惩罚贡献，不能直接读作米/弧度原始误差。

## 本次PPO实现

Actor/Critic均为512→256→256、ELU的前馈网络。扩散模型/参考固定，更新机器人控制策略、价值网络和动作分布噪声。每轮8192×24=196608条交互，GAE γ0.99、λ0.95；真实终止mask与超时价值bootstrap按实现处理。先建立回报目标，再整批标准化优势。

以新旧动作概率比构造裁剪代理目标（ε0.2），价值损失权重1.0、熵系数0.008；梯度范数上限1.0。裁剪目标不保证所有概率比都位于0.8—1.2，也不是KL硬约束。按KL目标0.01自适应学习率。

每批5个epoch、4个mini-batch，各49152条样本，每轮20次Adam更新。2000个PPO迭代对应约40000次优化器更新、393216000条环境转移。更新后重新采样，属于on-policy训练；冻结推理使用确定性均值动作、不继续更新参数/归一化统计。

原理来源：[PPO原始论文](https://arxiv.org/abs/1707.06347)。本次超参数、损失、优势计算和logger含义以服务器实际源文件及保存的agent.yaml为准。

## 冻结文件格式

`model_1999.pt`：PyTorch训练检查点字典，含Actor/Critic、优化器、迭代信息、normalizer，可恢复训练器。`exported/policy.pt`：由`torch.jit.script`导出的TorchScript Actor＋normalizer，是MuJoCo实际`torch.jit.load(...).eval()`加载格式；不含Critic/优化器/场景/参考数组，不能单独恢复完整PPO。

官方`play_eval.py`同时导出`exported/policy.onnx`。已核实六任务文件非空及SHA，但本次MuJoCo未使用ONNX后端，因此没有宣称ONNX执行等价性已验证。部署文件也需要任务接口JSON、参考NPZ、机器人MJCF/XML、动作映射、驱动和时间步。

| 任务 | 观测维数 | 输出动作维数 |
|---|---:|---:|
| Kick/Punch | 626 | 29 |
| Jump/Sit | 633 | 29 |
| Pick | 654 | 29 |
| Bimanual-Pick | 655 | 29 |

29维输出为27个身体关节动作与左右手各1个开合通道，经动作映射和执行器作用，不是直接关节力矩。接口维数来自已有Isaac导出快照；六个TorchScript SHA均匹配此前接口记录的policy SHA。本次读取文件/源码并计算哈希，没有启动新的仿真推理。

路径、大小、SHA和实际源码路径见`source_and_policy_format_evidence.json`。权重留在服务器，本次没有下载/上传权重。[Isaac Lab导出接口文档](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_rl/isaaclab_rl.html)仅用于通用格式说明，不替代本次源码核验。

## 本地报告

`D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929\实验结果汇报.html`

新增锚点：`#ppo-optimization-flow`、`#ppo-metric-priorities`、`#ppo-six-task-interpretation`、`#frozen-policy-file-formats`。

完整分析JSON、实际源码副本及页面验收保存在同目录下`论文复现汇报资料_20261002\PPO原理与策略导出`；原18个人体视频、28个机器人视频及质量筛查章节保留。界面修改脚本及只读证据收集脚本已保存到新项目，旧格斗仓库没有接收新报告修改。

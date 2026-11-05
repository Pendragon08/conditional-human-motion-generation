# DreamControl 原仓库仿真评估对齐

## 已完成的冻结 SuperMimic Pick 复测

按原仓库桌面、物体材质、质量扰动、物体重置和成功计数运行 100 个首回合。复用项目 29 身体关节加手资产，按参考首帧初始化，沿 x 平移 1.90 m 对齐机器人一侧的桌沿。位置仍从固定种子抽取的 100 条项目参考取得。物体重置高度 1.0 m，实测质量 0.10185–0.49728 kg。每回合 297 个控制步。100 个回合均到达交互点测量时刻，成功 0/100，右手接触 0/100。按项目固定时刻定义测得右腕误差 0.255011 m。首次按桌心平移的试跑在 21–62 步提前结束，保留记录并从结果中排除。

正式报告保存在 `D:/DreamControl_v2_upstream_pick_20261105/pick_100env_aligned.json`。原始脚本的成功条件是参考闭手时物体中心高于 0.95 m。项目手腕误差单列。原仓库的全播放窗口和机器人初始化没有在此适配中复刻。Pour、双手搬箱和推拉箱缺少同定义的公开任务配置，仍保留各自的项目协议评估。

## 初次核查

核对版本为 GenRobo/DreamControl 公开提交 `ec28b508313d6837de52c68247fbc8cf81fe4935`。本记录给出可复用入口、已确认的参数和评估差异。GPU 服务器 SSH 连接超时，本轮完成源码核对，实际仿真待验证。

## Pick 可复用的原仓库配置

- README 指定 `Training/scripts/reinforcement_learning/rsl_rl/play_eval.py` 与 `Isaac-Motion-Tracking-Pick-v0`。示例使用 1,000 个环境和 500 控制步视频窗口。与 v2 每任务 100 个随机化环境对齐时，须显式设置环境数，并记录固定种子和每个回合的边界。 [原仓库 README](https://github.com/GenRobo/DreamControl/blob/ec28b508313d6837de52c68247fbc8cf81fe4935/README.md#phase-2-rl-training)
- 直接复用 `G1PickEnvCfg` 的桌面、物体、物理属性和事件。代码中桌体尺寸为 1.0 × 2.0 × 0.8 m，物体尺寸为 0.05 × 0.05 × 0.20 m，基准质量 0.1 kg。启动事件在质量上加 (0.0, 0.4) kg 范围的样本。物体重置从参考库的抓取点及轨迹偏移得到平面位置，重置高度为 1.0 m，初始姿态为单位四元数，速度为零。保留原重置过程，并保存每次实测初始状态和 motion ID。 [Pick 场景及事件](https://github.com/GenRobo/DreamControl/blob/ec28b508313d6837de52c68247fbc8cf81fe4935/Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_pick_env.py) · [物体重置](https://github.com/GenRobo/DreamControl/blob/ec28b508313d6837de52c68247fbc8cf81fe4935/Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_interactive_base.py#L112)
- 原任务的计数在物体高度大于 0.95 m 且参考闭手标志为真时递增。`play_eval.py` 汇总整个运行窗口中至少成功一次的环境比例。原计数可原样报告。每环境第一回合的成功率单列，防止自动重置后的成功进入第一回合。原脚本输出运动记录和成功率，v2 交互点 L2 需新增有明确时间点、坐标系、目标点和缺失规则的旁路日志。当前项目已用的定时手腕误差继续保留为独立口径。 [原成功计数](https://github.com/GenRobo/DreamControl/blob/ec28b508313d6837de52c68247fbc8cf81fe4935/Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_interactive_base.py#L251) · [原汇总入口](https://github.com/GenRobo/DreamControl/blob/ec28b508313d6837de52c68247fbc8cf81fe4935/Training/scripts/reinforcement_learning/rsl_rl/play_eval.py)

## 接入冻结策略

先用原 Pick 环境和兼容的 RSL-RL 权重验证重置、动作与计数。公开播放入口按 RSL-RL 格式加载和导出策略。冻结 SuperMimic 的 29 身体动作、未来参考、历史观测和权重格式需通过独立推理接入层映射到原任务。使用逐关节名称验证、根状态坐标系和物体相对位置检查。对照评估由相同的场景配置、参考摆位或种子表、回合长度和打分器驱动。若保留当前 29 关节资产，在结果中标记原任务环境的机器人适配。冻结权重、日志路径与评估产物隔离，记录版本、种子、对象质量及位置、目标与实测腕点、回合终止状态。

## 四项任务的覆盖范围

| 当前任务 | 公开仓库可复用部分 |
|---|---|
| Pick | 原桌面 Pick 任务配置、重置与成功计数。按 29 身体关节冻结策略做接口适配。 |
| Pour | 可复用公共仿真及评估循环。原仓库任务注册未提供 Pour，v2 项目仓库当前是网页。v2 附录表 IX 的 0.65 m 是训练奖励门槛，完整倾倒与复正的评估判据、测试摆位和误差时刻仍待作者代码。 |
| 双手搬箱 | 原仓库 Bimanual Pick 为双手抬起任务。保留原任务作为独立对照。当前搬箱包含搬运和放下，需保留单独任务配置和阶段判据。 |
| 推拉箱 | 原仓库的 Open Drawer 是铰接抽屉。当前自由箱体前推、释放、再抓取、拉回使用单独配置。 |

v2 §IV-A 规定每项 100 个随机化仿真环境与 Success Ratio、Control Error 两项指标。附录 V-C 的位置与时序范围用于生成参考，完整仿真测试分布和交互点打分实现仍待作者发布。原仓库复用结果、29 关节适配结果及论文表 III 数值分栏报告。 [DreamControl-v2 评估设置](https://arxiv.org/html/2604.00202v1#S4.SS1) · [DreamControl-v2 项目仓库](https://github.com/GenRobo/DreamControl-v2) · [原仓库任务配置](https://github.com/GenRobo/DreamControl/tree/ec28b508313d6837de52c68247fbc8cf81fe4935/Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1)

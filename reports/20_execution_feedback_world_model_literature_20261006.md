# 执行反馈与世界模型：论文证据及候选方法（2026-10-06）

## 结论与研究边界

CLoSD 的执行反馈组织方式可以借鉴到 DreamControl，但不能直接复制骨架转换器。CLoSD 用实际执行的人体动作作为下一轮 DiP 的前缀；DreamControl 原作使用离线生成的参考动作训练任务策略。DreamControl 的 PPO 已经读取实际机器人状态，这一基础闭环不是新贡献。

“执行反馈”“动作条件动力学”“短时 MPC”均有正式论文先例。本项目先将它们登记为待验证候选，不宣称首次提出。候选贡献是：在冻结人体生成器及任务策略的条件下，验证任务保持的执行反馈与短时动力学辅助在 G1 运动跟踪中的作用及交互。

用户要求不再使用原始重定向后的额外机器人 IK。本轮保持该约束；原论文的 PyRoki 重定向保留。旧 I1 结果不能充当新方法的验证结果。

## CLoSD 与 DreamControl 的接口

| 方法 | 真实流程 | 证据与限制 |
|---|---|---|
| CLoSD，ICLR 2025 Spotlight | 文本/空间目标 → 冻结 DiP → 物理跟踪 → 实际运动缓冲 → HumanML3D 前缀 → 下一段生成 | 默认20个20Hz上下文帧、40个预测帧，约1秒上下文、2秒计划。控制器30Hz；微调3072环境/4000轮/CFG2.5，评估4096环境/CFG7.5。 |
| DreamControl，作者官网标注 ICRA 2026 | 文本/空间提示 → 已发布 OmniControl → 人体动作 → G1重定向/筛选 → 固定参考 → 任务PPO | 模拟跟踪 actor 读取机器人状态与当前/未来参考。实机无参考观察变体经过另行训练；不能与模拟策略混写。 |

CLoSD `closd.py` 的 `_update_pose_buffer()` 从 `_rigid_body_pos` 采集实际动作；`build_completion_input()` 经过 `pose_to_hml()` 后写入 `model_kwargs['y']['prefix']`。论文§4.3明确微调时固定 DiP。DreamControl `collect_Kick.sh` 则依次调用离线生成、retarget、refine；审计的公开路径未识别出执行轨迹回写 OmniControl，亦未识别出学习状态转移的世界模型。

来源：[CLoSD 原文§4.1–5.1](https://arxiv.org/html/2410.03441v1)、[CLoSD 固定源码](https://github.com/GuyTevet/CLoSD/blob/d8950bdb78312fcd2844f336b3fc079e7368c87f/closd/env/tasks/closd.py#L199)、[DreamControl 原文§III–IV](https://arxiv.org/html/2509.14353v3)、[DreamControl 生成流水线](https://github.com/GenRobo/DreamControl/blob/ec28b508313d6837de52c68247fbc8cf81fe4935/TrajGen/collect_Kick.sh)、[DreamControl 作者官网](https://genrobo.github.io/DreamControl/)。

CLoSD 的24点 SMPL-compatible角色与22点/263维 HumanML3D有专门转换。G1是27个身体关节角、root位姿及手状态，肢长、关节索引和坐标约定不同。G1 FK坐标不能直接冒充HumanML前缀。稀疏执行提示与完整实际动作前缀是两种方法，应分开测试。

注意：CLoSD的`context_switch_prob=1`只替换前缀内容，目标坐标转换和重建对齐仍可能来自实际姿态。这不是自动取消全部生成器反馈的开关。

## 正式发表的相关工作

以下以正式会议/期刊记录与作者论文为证据；模拟角色与实体机器人明确区分。

| 论文 | 正式出处/标识 | 世界模型的作用 | 可借鉴部分及证据边界 |
|---|---|---|---|
| SuperTrack: Motion Tracking for Physically Simulated Characters using Supervised Learning；Fussell等 | ACM TOG40(6)，SIGGRAPH Asia2021；10.1145/3478513.3480527 | 从状态及PD目标预测加速度，积分成后继状态；通过可微模型训练跟踪策略 | 短rollout、相对根状态、PD残差。长预测漂移/滑脚；仅仿真角色。[作者PDF](https://theorangeduck.com/media/uploads/other_stuff/SuperTrack.pdf) |
| Physics-based Character Controllers Using Conditional VAEs；Won等 | ACM TOG41(4)，SIGGRAPH2022；10.1145/3528223.3530067 | 专家数据训练并冻结后继状态模型；状态一致性损失向CVAE控制器反传 | 明确动力学层开/关消融；下游主要DD-PPO，不等同在线MPC；仅仿真。[作者论文及下载](https://ai.meta.com/research/publications/physics-based-character-controllers-using-conditional-vaes/) |
| ControlVAE；Yao等 | ACM TOG41(6)，SIGGRAPH Asia2022；10.1145/3550454.3555434；arXiv2210.06063 | 动作条件速度增量，交替真实仿真replay与模型训练；支持MPC及模型反传 | 多步监督8帧、控制器24帧；单步训练可发散。新增物体/其他角色是限制；仅仿真。[原文](https://arxiv.org/html/2210.06063) |
| MoConVQ；Yao等 | ACM TOG43(4)，SIGGRAPH2024 Journal Track；10.1145/3658137；arXiv2310.10198 | 可微动力学与离散VQ运动表征交替训练 | VQ代码不是世界模型；主要消融是VQ结构。稀有动作、未来代码延迟、单人环境限制。[官方代码](https://github.com/heyuanYao-pku/MoConVQ)、[正式TOG目录](https://www.siggraph.org/wp-content/uploads/2024/09/tog-43-4.html) |
| Puppeteer: Hierarchical World Models as Visual Whole-Body Humanoid Controllers；Hansen等 | ICLR2025；arXiv2405.18418 | 两层TD-MPC2，预测潜状态、奖励、终止、Q值；MPC跟踪身体关键点 | 与运动跟踪最接近；56DoF DMControl仿真人物，无实体人形。正式版10seeds；低层10M步，不是轻量即插即用。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/9b867f0e56c4c085ef1cfdad691db5f6-Paper-Conference.pdf) |
| TD-MPC2；Hansen等 | ICLR2024；arXiv2310.16828 | 无观测解码器的潜动力学/奖励/Q/策略先验，短时采样MPC | 原2024论文无done head；通用仿真控制证据，不直接证明G1动作跟踪。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/cf73d57b6dcda32b293df7c2d5341f49-Paper-Conference.pdf) |
| Mastering diverse control tasks through world models（DreamerV3）；Hafner等 | Nature640，647–653，2025；10.1038/s41586-025-08744-2 | RSSM预测潜状态、观测、奖励、continue；想象轨迹训练actor/critic | 部署用actor，无在线MPC；不同agent共享算法配置，不是统一共享权重；无实体人形验证。[Nature原文](https://www.nature.com/articles/s41586-025-08744-2) |
| DWL；Gu等 | RSS2024；10.15607/RSS.2024.XX.058；arXiv2408.14472 | 历史本体观测去噪，重建当前完整状态供PPO | 实体XBot-S/L、手臂固定。主要是当前状态估计，不能当成动作条件未来预测/MPC。[RSS正式记录](https://www.roboticsproceedings.org/rss20/p058.html) |
| HAIC；Li等 | RSS2026；10.15607/RSS.2026.XXII.013；arXiv2602.11758 | 本体历史+未来参考估计当前隐藏物体状态，几何特征供PPO | 实体G1物体交互；23动作维，排除6腕自由度。隐藏物体估计可能向参考漂移；不是候选动作rollout MPC。[RSS正式记录](https://www.roboticsproceedings.org/rss22/p013.html)、[作者修订PDF](https://haic-humanoid.github.io/static/haic.pdf) |

DreamControl-v2（arXiv2604.00202，本轮仅核实预印本）已经提出机器人空间文本/时空引导运动先验。单凭“在G1空间训练扩散”也不能声称首创。[预印本方法](https://arxiv.org/html/2604.00202v1)

## 本轮选择

优先借鉴ControlVAE/SuperTrack的短时动作条件预测与多步监督；参考Puppeteer的短时预测辅助，但第一轮不直接复制其双层TD-MPC2的大预算。DWL/HAIC的状态估计路线另列为未来物体任务对照，不与预测动力学混称。

候选F：任务保持的执行反馈重规划。先验证稀疏root/末端反馈映射；完整HumanML前缀另设门控。

候选W：短时动作条件动力学辅助。先评估留出参考上的预测，再用同候选预算控制对照判断是否带来实际收益。预测loss下降不等于任务成功提升。

两项分别通过后再运行F×W四格实验。严格消融协议见 `plans/feedback_world_model_ablation_20261006.md`。本文件是调研与设计，未以引用论文的成绩代替本项目实验。

# 人形机器人模仿学习应用场景与开源文献调研

检索日期：2026-10-08。研究对象：条件人体动作生成 → 原始 G1 重定向 → Isaac Lab 中的 PPO 运动跟踪 → 物理执行 → MuJoCo 跨仿真验证。

本轮核对了 16 篇正式发表的相关或邻近方向论文，12 项提供公开代码地址。核验依据为期刊/会议记录、作者项目页和官方仓库；“有公开实现”表示已发现实际源码及使用说明，本轮未安装或运行这些外部实现。部分代码限制商业使用，代码、模型和数据许可应分别核对。以下应用选择和移植建议属于依据文献作出的研究判断，不表示我们的系统已经实现对应能力，也不表示执行反馈或世界模型创新点已经验证有效。

## 1. 最适合当前路线的应用

| 应用方向 | 可以设计的具体任务 | 文献已展示的证据 | 我们需要补充的部分 |
|---|---|---|---|
| 家庭与服务场景交互 | 开门/抽屉、低位取物、物品交接、放到指定台面 | HumanPlus 有自主整理/家务任务；HOMIE 和 TWIST 有真实 G1 遥操作取放及门类交互 | 物体与场景感知、手部动作、实际接触、成功后状态验证 |
| 仓储与大件搬运 | 双臂拾箱、抱持、搬运、货架转移、放置 | HumanPlus 有货架卸物；HOMIE 有箱体货架转移；Punyo 有真实上半身大件操作 | 箱体位姿/质量、负载下平衡、稳定抓持、手臂/躯干接触和防滑落 |
| 全身遥操作与示范采集 | 远程作业、全身示教、人在回路的移动操作 | TWIST 为真实 G1 全身模仿遥操作；OmniH2O 为真实 H1 多输入控制 | 第一视角回传、手部/触觉接口、网络延迟和接触反馈；危险现场巡检是应用推断，非上述论文直接验证 |
| 自然动作与服务表达 | 招手、指引、蹲下、舞蹈、风格化行走 | SONIC 为真实 G1 通用跟踪，支持文本/视频/VR等控制；ASAP 展示动态全身技能 | 语义与动作一致性、实时生成延迟、动作连续性、长时间稳定性 |
| 体育与协调技能 | 跳跃、单脚平衡、运动示教、手脚协调动作 | ASAP 有真实 G1 敏捷动作；ZEST 有多机器人高动态动作 | 时间尺度与执行器限制、落地冲击、困难片段训练；完整运动项目还需要球/物体感知及真实接触任务 |
| 物理人体与交互数据生成 | 生成带物体的人体示范、修复接触、产生机器人训练参考 | InterAct 提供生成与数据处理；InterMimic 提供物理交互模仿 | 从人体交互到 G1 的形态适配和物理执行；生成质量、仿真角色质量与真实机器人成功须分开统计 |

对应主源：[HumanPlus](https://proceedings.mlr.press/v270/fu25a.html)、[HOMIE](https://www.roboticsproceedings.org/rss21/p070.html)、[TWIST](https://xbpeng.github.io/projects/TWIST/index.html)、[Punyo](https://doi.org/10.1126/scirobotics.ads6790)、[OmniH2O](https://proceedings.mlr.press/v270/he25b.html)、[SONIC](https://nvlabs.github.io/GEAR-SONIC/)、[ASAP](https://www.roboticsproceedings.org/rss21/p066.html)、[InterAct](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_InterAct_Advancing_Large-Scale_Versatile_3D_Human-Object_Interaction_Generation_CVPR_2025_paper.html)、[InterMimic](https://sirui-xu.github.io/InterMimic/)。

## 2. 优先阅读且具有公开实现的论文

| 论文与作者 | 正式发表 | 方法及证据 | 公开实现与适配边界 |
|---|---|---|---|
| **HumanPlus: Humanoid Shadowing and Imitation from Humans**，Fu 等 | [CoRL 2024](https://proceedings.mlr.press/v270/fu25a.html) | 低层 PPO 模仿人体动作，高层视觉行为克隆学习自主技能；真实定制双足人形 | [代码](https://github.com/MarkFzp/humanplus)：跟踪、视觉模仿与部署；不是 G1，原框架为 Isaac Gym |
| **OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning**，He 等 | [CoRL 2024](https://proceedings.mlr.press/v270/he25b.html) | PPO 教师/学生跟踪，可连接文本条件动作生成和遥操作示范；真实 H1 | [代码](https://github.com/LeCAR-Lab/human2humanoid)：重定向、训练与部署；高层 Diffusion Policy 引用外部实现，不能声称全栈一键开源 |
| **HOVER: Versatile Neural Whole-Body Controller for Humanoid Robots**，He 等 | [ICRA 2025](https://ieeexplore.ieee.org/document/11128549/) | 运动模仿 RL oracle，经命令掩码和 DAgger 蒸馏为多控制模式策略；真实 H1 | [代码](https://github.com/NVlabs/HOVER)：Isaac Lab 训练、MuJoCo 验证与 H1 部署；适合参考当前训练平台，G1 需另行适配 |
| **HOMIE: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit**，Ben 等 | [RSS 2025](https://www.roboticsproceedings.org/rss21/p070.html) | 无 MoCap 先验的低层 RL 加同构外骨骼遥操作；采集数据用于模仿学习；真实 G1 | [代码](https://github.com/InternRobotics/Homie)：训练、硬件与部署；复杂任务多为遥操作，低层不是人体运动模仿，CC-BY-NC-SA-4.0 |
| **TWIST: Teleoperated Whole-Body Imitation System**，Ze 等 | [CoRL 2025](https://openreview.net/pdf?id=htgNQHa6Ta) | 人体动作重定向、PPO 教师、RL/BC 学生及末端扰动；真实 G1 接触任务遥操作 | [代码](https://github.com/YanjieZe/TWIST)：数据、训练、权重和 sim-to-sim/real；原训练为 Isaac Gym，自主生成动作替代操作者尚需验证 |
| **ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills**，He 等 | [RSS 2025](https://www.roboticsproceedings.org/rss21/p066.html) | PPO 跟踪预训练，再用目标动力学数据学习残差动作并微调；真实 G1 | [代码](https://github.com/LeCAR-Lab/ASAP)：训练、对齐、部署及 MuJoCo；主要相位单动作策略，不能当任意参考通用跟踪器 |
| **SONIC: Supersizing motion tracking for natural humanoid whole-body control**，Luo 等 | [Science Robotics 2026](https://doi.org/10.1126/scirobotics.aed4592) | 大规模人体动作模仿，统一动作接口连接规划、遥操作及高层模型；真实 G1 | [代码](https://github.com/NVlabs/GR00T-WholeBodyControl)：Isaac Lab PPO、微调、权重、部署和 MuJoCo；公开数据只是论文数据的一部分，优先复用权重，小预算适配 |
| **Learning contact-rich whole-body manipulation with example-guided reinforcement learning**，Barreiros 等 | [Science Robotics 2025](https://doi.org/10.1126/scirobotics.ads6790) | 示范引导 RL 学习手臂/躯干多部位接触；真实 Punyo 操纵大件 | [代码](https://github.com/ToyotaResearchInstitute/Punyo_EGRL)：Isaac Gym 任务、训练与策略播放；是柔顺机器人上半身，非走动双足全身 |
| **InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions**，Xu 等 | [CVPR 2025 Highlight](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_InterMimic_Towards_Universal_Whole-Body_Control_for_Physics-Based_Human-Object_Interactions_CVPR_2025_paper.pdf) | 接触感知模仿、教师/学生与物理修正示范；论文主体为仿真人体 | [代码](https://github.com/Sirui-Xu/InterMimic)：Gym 训练及当前 G1 仿真、Lab 回放/推理；不代表已完成 G1 真机或等价 Lab 训练 |
| **InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation**，Xu 等 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_InterAct_Advancing_Large-Scale_Versatile_3D_Human-Object_Interaction_Generation_CVPR_2025_paper.html) | 文本条件人—物交互生成、接触质量修正和仿真数据接口 | [代码](https://github.com/wzyabcas/InterAct)：生成训练/推理、接触引导和资产转换；属于生成器/数据层，须接物理控制器 |
| **Hierarchical generative modelling for autonomous robots**，Yuan 等 | [Nature Machine Intelligence 2023](https://www.nature.com/articles/s42256-023-00752-z) | 模拟 Valkyrie 的分层规划、移动与操作；移动部分含 MoCap 模仿 | [代码](https://github.com/Yunaik/hgm4robots)：高/中/低层实现、训练与策略；论文明确物理仿真证据，非真机部署；generative model 不是扩散模型 |
| **Learning Humanoid Standing-up Control across Diverse Postures**，Huang 等 | [RSS 2025](https://www.roboticsproceedings.org/rss21/p064.html) | 无动作参考的 PPO 起身与恢复；真实 G1 | [代码](https://github.com/InternRobotics/HoST)：Gym 训练/评估；可作恢复分支，不能归类为人体模仿方法，部分联合训练未完成 |

CoRL 2024 的 HumanPlus、OmniH2O 等论文属于 2024 年会议，但 PMLR 270 论文卷出版年份标注为 2025。会议年和出版年分别保留，避免误写为 CoRL 2025。

这些方法并非都使用扩散模型，也不是全部端到端自主。人体参考跟踪、从机器人示范学习任务策略、无先验 RL 以及人在回路的遥操作，分别解决不同层级问题。已有全身跟踪器不会自动具备物体识别、抓握或长任务规划能力。

## 3. 相关方向，但尚未核实可直接使用的官方实现

| 论文 | 正式发表 | 能支持的应用判断 | 本轮核验限制 |
|---|---|---|---|
| Sleiman 等，**ZEST: Zero-shot embodied skill transfer for athletic robot control** | [Science Robotics 2026](https://doi.org/10.1126/scirobotics.aec7695) | 人体/视频/动画参考，经 Isaac Lab PPO 学习高动态及多接触技能，包含真机 | [作者资料](https://rai-inst.com/resources/in-the-press/zest-zero-shot-embodied-skill-transfer-for-athletic-robot-control/)；未发现官方训练/部署仓库 |
| Dao 等，**Sim-to-Real Learning for Humanoid Box Loco-Manipulation** | [ICRA 2024](https://ieeexplore.ieee.org/document/10610977/) | 真实 Digit 搬箱的物体任务奖励与负载平衡参考 | [论文](https://arxiv.org/abs/2310.03191)；未发现官方仓库；方法为 RL，非人体示范模仿 |
| Zhuang 等，**Humanoid Parkour Learning** | [CoRL 2024](https://proceedings.mlr.press/v270/zhuang25a.html) | 真实 H1 深度感知越障、台阶与间隙通过 | [项目](https://humanoid4parkour.github.io/)；未发现官方人形代码；无需人体先验的 RL，不能拿四足前作代码替代 |
| Lin 等，**Let Humanoids Hike! Integrative Skill Development on Complex Trails** | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_Let_Humanoids_Hike_Integrative_Skill_Development_on_Complex_Trails_CVPR_2025_paper.html) | 地形感知、导航和底层控制联合学习 | [项目](https://lego-h-humanoidrobothiking.github.io/)仍标代码待发布；主要仿真，不是户外真机验证 |

复杂地形移动具有应用价值，但需要加入地形感知、落脚规划和障碍物课程；当前只有文本/空间目标与人体参考时，不能直接推定能够在台阶或复杂现场稳健执行。医疗康复或照护等应用亦不能仅凭模仿动作演示推断疗效或临床可用性，本轮不作为直接落地方向。

## 4. 对现有研究路线的具体选择

**优先选择：物体与接触条件下的家务/物流全身操作。** 建议逐步从“开抽屉/门”“低位拾箱并抬离地面”“搬箱后放到指定台面”扩展。我们已有对应动作和控制任务基础，新增工作集中在真实物理目标、接触和闭环协调，较容易形成清楚的实验问题。这是结合现有项目基础作出的建议，不是文献已经证明我们的方案成功。

生成端可借鉴 InterAct：在文本与空间约束外加入物体几何、起始/目标位姿和接触阶段，同时生成身体及物体相关轨迹。物理端可借鉴 InterMimic/Punyo：以参考模仿为基础，同时优化物体任务、接触一致性、平衡和动作代价。G1 跟踪实现可参考 TWIST，或优先研究 SONIC 的已发布 G1/Isaac Lab 实现；HOVER 提供另一条明确的 Lab→MuJoCo 控制研究路径，但原真机为 H1。

执行反馈可以让后续参考依据实际机器人和物体状态重新生成；世界模型可以研究预测接触、物体状态和失稳趋势，以辅助 PPO 的控制优化。以上是待验证方法，不能将“加入模型”本身视为改善。尤其当前项目的反馈/世界模型实验尚未证明总体有效，应保留稳定控制基线并使用同参考、同任务和匹配训练预算的对照。

验证顺序仍为：**Isaac Lab 先建立实际接触任务和训练跟踪策略 → 冻结策略 → MuJoCo 中保留对应物体、碰撞和任务条件做 sim-to-sim → 再考虑真机。** 主指标使用门/抽屉实际开度、物体抬升与稳定保持、放置位置误差、跌倒率和任务成功率；参考位置接近和机器人姿态相似只是辅助指标。扰动、物体质量/摩擦变化和未见目标位置用于检验泛化，不能用单段好看视频替代统计。

本轮未启动训练、评估或修改既有实验结果。结构化核验目录见同仓库 results/literature/humanoid_imitation_applications_20261008.json。

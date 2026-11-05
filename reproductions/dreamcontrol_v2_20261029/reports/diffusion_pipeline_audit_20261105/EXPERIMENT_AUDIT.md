# DreamControl-v2 条件扩散训练与 2,000 条参考轨迹审计

2026-11-05 · 同模型家族的独立 Codex（GPT-6 系列）审查 · 临时结论（provisional）

结论 **WARN**。已有项目评估文件与公布数值吻合。确认一处采样高度坐标约定不一致、与固定公开源码不同的空间引导、以及高权重后优化造成的配对动作分布偏移。以下文件路径相对于本复现目录。只读检查已有结果、公开源码及静态 G1 正向运动学，未重训或重采样。SSH 连接超时，服务器端数据和模型身份沿用既有哈希绑定收据，源码比较通过 GitHub 固定提交完成。

## 原 DreamControl-v2 训练了多少步

- **公开材料没有披露作者的优化器更新次数，不能确认是 10,000。** v2 §III-B 说明使用 MDM 初始化 OmniControl、预测干净信号，§IV-A 和表 I 报告训练数据混合及规模，公开项目仓库提供网页材料。表 I 的 21.8k、26.4k、29.6k 是不同数据混合的样本数量。 [v2 论文 §III-B](https://arxiv.org/html/2604.00202v1#S3.SS2) · [v2 论文 §IV-A](https://arxiv.org/html/2604.00202v1#S4.SS1) · [v2 网页仓库](https://github.com/GenRobo/DreamControl-v2)
- 固定原 DreamControl 的 Pick 公开入口直接加载现成 OmniControl 人体模型采样，再重定向与优化。OmniControl 论文附录 A.2 报告其人体模型训练 **250,000 次迭代**，batch 64、AdamW lr 1e-5。其公开训练 README 另给 `--num_steps 400000` 命令示例，从 `model000475000.pt` 初始化。公开实现用该参数推算整 epoch 数，检查点命名还会加上从文件名解析出的已有步数。这些值描述人体模型及其公开脚本，不能确定 v2 的机器人微调预算。 [原 DreamControl Pick 入口](https://github.com/GenRobo/DreamControl/blob/ec28b508313d6837de52c68247fbc8cf81fe4935/TrajGen/collect_Pick.sh#L2) · [OmniControl 论文 A.2](https://arxiv.org/html/2310.08580#A1.SS2) · [OmniControl 官方训练命令](https://github.com/neu-3dvi/OmniControl#train-your-own-omnicontrol) · [公开训练循环](https://github.com/neu-3dvi/OmniControl/blob/main/train/training_loop.py#L43)
- 本项目的 **10,000 是实际使用的机器人先验检查点的优化器更新数**，microbatch 16、累计 4、AdamW lr 1e-5。1000 是 cosine DDPM 的扩散时间步。两份正式批采样器校验同一 10,000 步 SHA-256 `109fcfebf2a805831444ca8fb55d69674ef7cb19c61219e3e4ed18876d9527a9`、数据来源和步数。证据为 `reports/stage2_prior29_20261103/config.json:12–38`、`src/stage2_robot_prior.py:127–135,374–391,403–441`、`src/stage3_sample_pick29_batch.py:45–51`、`src/stage3_sample_supported29_batch.py:44–53`。

## 配对结果和因果证据

| 技能 | FID 原始 → 后优化 | 右腕或双腕全条件点 L2 原始 → 后优化 | Diversity 原始 → 后优化 |
| --- | ---: | ---: | ---: |
| Pick | 1.5988 → 3.7860 | 37.43 cm → 0.237 cm | 1.0847 → 0.9148 |
| Pour | 2.1266 → 6.4640 | 29.90 cm → 0.298 cm | 1.0019 → 0.4708 |
| 双手搬箱 | 0.7038 → 4.9120 | 36.34 cm → 0.164 cm | 1.1896 → 0.4144 |
| 推拉箱 | 2.4662 → 5.4430 | 49.62 cm → 0.144 cm | 1.1669 → 0.9702 |
| 合并 | 1.4175 → 4.3705 | 38.32 cm → 0.211 cm | 1.3325 → 1.3768 |

腕点数字是按轨迹平均的条件误差。包含根、膝或脚的 Control L2 从 0.33411 m 降至 0.02991 m，公开脚滑判据从 7.32% 降至 0%。项目对照池 Diversity 为 1.9940。合并多样性可能含任务之间差异，各任务内的多样性均下降。四类 500 条皆为最终入选样本，配对比较固定入选 ID。其结果没有估计完整候选池的原始模型指标或多训练种子方差。证据为 `reports/generated_reference_evaluation_20261105/summary.md:3–32` 和 `geometry.json` 中 `tasks.*.*.wrist_only_control_l2_m`。

### 1. 后优化改变分布，配对重编码已经排除主要格式解释

相同 256 个 ID、相同 586 条对照、匹配 T−1 长度，原 263D FID **1.6704**，从原始关键点按同流程重编码 **1.6543**，最终 FK 关键点按同流程重编码 **4.4243**。每个技能均存在最终相对同流程重编码的 FID 升高。本诊断直接定位差距出现在原始点列到最终机器人点列的处理阶段，无法单独分解根、脚、腕约束各自的效应。证据为 `D:/DreamControl_v2_upstream_pick_20261105/fid_roundtrip_256.json:69–218`。

实际多数样本走批量求解器。Pick 以及三项批量求解器都把 decoded 22 点拟合残差乘 3，把同一时刻腕点目标残差乘 100 或 300，根位姿固定模板残差乘 100，踝点固定模板残差乘 80 或 240，推拉 v7 小样踝点残差乘 480。最小二乘会再平方这些乘数。腕姿态还施加贯穿时间的目标与插值，关节和根从物理落地模板求解，关节逐帧平滑。多数候选的原始腕点离目标数十厘米，这些项相对于保持原动作的拟合有足够强的拉力，实测腕点控制和脚滑改善与按任务 FID、Diversity 变差同时出现。合并清单显示双手搬箱与推拉各 498 条走 bulk、各两条走试点，Pour 498 条走 bulk（包括两条见证）和两条走 v4，Pick 来自数次既有队列。证据为 `src/stage3_pick500_postopt.py:63–110,122–144,172–185`、`src/history_stage3_pick500_postopt_follow_v3.py:65–128`、`src/stage3_supported_bulk_postopt_v1.py:66–109,121–159,169–176`、`src/stage3_push_task_postopt_v7.py:75–95` 和 `reports/four_task2000_final_20261105/native_motion_manifest.json` 的 `source` 字段。

论文允许机器人空间后优化和脚稳定、避障、腕方向检查，Pick 附录给出手腕、SDF 与平滑代价。项目固定站姿、全程腕目标和损失乘数是自行设定的实现，作者未公开这些配方。将腕点到达、二进制闭手计划和模拟物体接触分别验收。 [v2 §III-B4 与附录 V-D](https://arxiv.org/html/2604.00202v1#S5.SS4)

### 2. 训练编码与空间采样使用了不同高度基准

`src/feature263.py:22–36,47–57` 先把世界 Z-up 转为 Y-up，减去每段 22 点的全局最低高度，再消去首帧骨盆水平位置及旋转。两份任务采样器只使用轴置换，保存的平移是 `[0,0,0]`，将场景绝对 z 直接当成模型空间 y。后优化将原点列和目标都用同一零平移逆变换回世界。矩阵的轴和手性符合这套映射，问题出在高度零点。证据为 `src/stage3_sample_pick29_batch.py:54–88,154–159`、`src/stage3_sample_supported29_batch.py:56–82,148–155`、`src/stage3_supported_bulk_postopt_v1.py:136–143`、`src/stage3_pick500_postopt.py:135–141`。

只读计算固定提交的 URDF 正向运动学，取两份脚本共同的初始姿态及世界根高 0.84 m。22 点中的最低值是左右踝滚转连杆原点，z = **0.0970771748 m**。把这同一姿态当作静止训练片段，编码的踝点 y = **0**、骨盆 y = **0.7429228252 m**。直接采样的模板踝点 y = **0.0970771748 m**、骨盆 y = **0.84 m**。对这个姿态，两坐标系差 **9.71 cm**。这不等于所有轨迹相差固定 9.71 cm，也不足以解释全部 30–50 cm 腕误差。后优化以实际碰撞几何最低点计算站姿根高，进一步说明连杆最低点与物理地板是两个基准。证据为 `src/g1_fk.py:13–20`、`src/stage3_pick500_postopt.py:63–68`、`src/stage3_supported_bulk_postopt_v1.py:63–69` 及本报告 JSON 的 FK 见证。 [固定 G1 29DoF URDF](https://github.com/GenRobo/DreamControl/blob/ec28b508313d6837de52c68247fbc8cf81fe4935/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf)

Pick 的 4/5 是左右膝，论文 Pick 明确加双膝约束。三项采样的 10/11 是踝滚转连杆，20/21 是左右腕偏航连杆，符合表 VII。实际 Pick 的明确偏离是缺少肘点，接触固定在帧 40，论文写 40–50 采样，并使用不同工作空间及明确右手的文字。项目四项采用 120/180 帧、单项单固定文字及手工腕点时序。 [v2 附录 V-A 与 V-C](https://arxiv.org/html/2604.00202v1#S5.SS3)

### 3. 训练与生成没有沿用固定公开源码的完整空间引导

- 原 DreamControl `TrajGen/diffusion/gaussian_diffusion.py:423–497,527–547,696–719` 在训练 `q_sample` 后调用空间引导。训练每次 20 步，按该代码的首样本时间步判据在低噪声时用 100 步。采样先算 posterior mean，再做逐目标欧氏距离之和的梯度更新，常规 10 步、最后 10 个时间步每次 500 步，并按每关节最大关键帧数缩放。 [固定源码](https://github.com/GenRobo/DreamControl/blob/ec28b508313d6837de52c68247fbc8cf81fe4935/TrajGen/diffusion/gaussian_diffusion.py#L423)
- 实际训练调用 `src/stage2_robot_prior.py:180–189`，加噪后直接预测 x0，学习文本与 ControlNet 式空间分支，未对输入加噪样本做梯度空间修正。两份正式采样器在同一个 posterior mean 接入点覆写 `guide`，所有时间步均用两次按条件坐标数平均的 MSE 更新，方差下界 .01、tau 1000、归一化通道步长截断 .25、六次回退。递减同一步的目标函数不能证明全程最终控制达标。MSE 符合 v2 §III-B3 的文字，但论文未公布两步、回退或训练 guidance 的对应实现，差异只能确证到固定公开基线。证据为 `src/stage3_sample_pick29_batch.py:95–141`、`src/stage3_sample_supported29_batch.py:89–135`、`reports/stage2_prior29_20261103/config.json:33–38`。 [v2 §III-B3](https://arxiv.org/html/2604.00202v1#S3.SS2.SSS3)
- 训练每条随机 1–2 个点、随机帧且 10% 清空空间提示。采样同时给 4–5 个点，根和任务腕全程约束，其余每 5 帧。该任务条件组合从未以相同关节点数出现在训练。项目与公开源码都将整条首帧特征梯度置零，连带首个速度增量和首帧局部位置。其继承行为与密集提示、按条件点平均和无末端额外迭代的相对贡献尚未做隔离。证据为 `src/stage2_robot_prior.py:148–176`、`src/stage3_sample_pick29_batch.py:82–94,109–126`、`src/stage3_supported_task_conditions.py:82–84`。
- 训练数据的归一化只累计训练分区并沿用相同哈希，采样按同一统计反归一化。已检查噪声时间步范围、x0 目标、归一化/反归一化维度、时间 padding 和显式点掩码，未见广播或左右点错位。文本 CFG 2.5 走公开 wrapper，文本去条件分支仍保留空间提示。模型真实入口是 `stage2_robot_prior.py` 和 `stage2_cmdm29.py`，旧实验 `train_robot_prior.py`、`robot_motion_dataset.py`、`robot_training_step.py` 不是这个检查点的调用链。证据为 `src/prepare_grab_omni_mixed29_full_preview_20261103.py:88–108`、`src/package_mixed29_weak_baseline_20261103.py:39–41`、`src/stage2_robot_prior.py:73–123,127–189`、`src/stage2_cmdm29.py:19–43,66–101`。 [公开 CFG 包装器](https://github.com/GenRobo/DreamControl/blob/ec28b508313d6837de52c68247fbc8cf81fe4935/TrajGen/model/cfg_sampler.py#L25)
- 2,091 条训练样本来自 GRAB/DexWeave 903 与 OmniRetarget/OMOMO 1,188，合计 3,161 条含验证和测试。513 条裁剪样本保留整段源描述，当前训练文字中 Pour 直接描述仅 16 条。模型以 MDM 人体主干及其复制控制主干热启动，控制输出层为零，CLIP 冻结。项目用 29 身体关节，论文后优化及部署用 27，Full-Mix 同时包含 AMASS 和 Nymeria。训练验证只打固定随机加噪下的去噪 MSE，不能证明 10,000 步充分或不足，也不能由 MSE 推断接触/长时任务能力。证据为 `reports/grab_omni_29dof_paper_scope_completion_20261103.json:4–38`、`reports/grab_omni_29dof_package_audit_20261103.json:5–13`、`reports/current_prior_task_coverage_20261104/任务覆盖与范围说明.md:3–17`、`src/stage2_cmdm29.py:71–101`、`src/stage2_robot_prior.py:192–205,435–441`。 [v2 表 I 和训练混合](https://arxiv.org/html/2604.00202v1#S4.SS1)

## 评估完整性 A–F

| 检查 | 结论 | 证据及解释 |
| --- | --- | --- |
| A 对照来源 | PASS | FID 对照为 586 条数据集留出的真实来源机器人特征，来自 323 个名义来源动作。评估检查组、源 ID 和特征哈希没有跨分区交集，逐文件校验。数据来自重定向的观测/遥操作数据，域外独立性与标签逐条语义未核证。Control L2 则比较预先保存的空间命令，属于条件约束测量。证据为 `src/evaluate_generated_reference_embeddings_20261105.py:101–106,123–138`、`src/evaluate_generated_reference_geometry_20261105.py:52–84`。 |
| B 分数归一化 | PASS | 项目训练统计与评估器的 HumanML3D 统计用途各异。评估器在原始 263D 上用固定公开均值/方差，所有组用相同归一化。FID 由原始嵌入协方差计算，L2 直接用米，分母是观测数量。公开 wrapper 也忽略最后四个脚接触通道。证据为 `src/evaluate_generated_reference_embeddings_20261105.py:45–99,150–195`。 |
| C 结果存在 | PASS | 核对 `geometry.json`、`embedding_metrics.json` 哈希分别为 `3e632cb0…09141f`、`5bc0d785…b4cc85`，与 `receipt.json:2–8` 一致。最终清单各 500、候选 ID、初始噪声哈希、特征内容哈希和最终状态指纹均有 2,000 个不同值。摘要数值可由完成 JSON 读取。 |
| D 代码执行 | PASS（项目） | 当前五项有真实计算调用和结果。固定公开评估入口中 R-Precision/Matching 仅有占位的零，公开 FID 函数在同一循环第二次写入不同参考的值。项目直接复用指标函数并自行定义检索，未采用这些旧入口输出。证据为 `src/evaluate_generated_reference_embeddings_20261105.py:150–205,208–261`、`src/evaluate_generated_reference_geometry_20261105.py:76–97`。 |
| E 声明范围 | WARN | 2,000 条是单检查点、固定四项、最终通过筛选后的配对样本。FID 使用共享 GRAB+Omni 对照，61 条对照裁至 196 特征帧。R-Precision 是四条重复任务提示分别对 31 条弱标签文本，且只有相同原文才排除，检索难度依赖干扰池，文字为零的两项不能单独归因到模型。不能与论文 Full-Mix 表 I 做同口径数值比较。 |
| F 评估类型 | PASS（分类） | FID 为 real_gt 分布对照，Diversity、几何脚滑及目标遵守为 self_supervised_proxy，R-Precision 是固定指令与干扰词构成的检索代理，物体成功率是单列 simulation_only。两份生成评估器均未把候选从模型输出反制成真实对照。 |

[固定公开匹配评估入口](https://github.com/GenRobo/DreamControl/blob/ec28b508313d6837de52c68247fbc8cf81fe4935/TrajGen/eval/eval_humanml.py#L20) · [固定公开 FID 入口](https://github.com/GenRobo/DreamControl/blob/ec28b508313d6837de52c68247fbc8cf81fe4935/TrajGen/eval/eval_humanml.py#L172) · [固定公开匹配网络](https://github.com/GenRobo/DreamControl/blob/ec28b508313d6837de52c68247fbc8cf81fe4935/TrajGen/data_loaders/humanml/networks/evaluator_wrapper.py#L66)

腕点的 3D 位置与腕朝向、掌指接触是不同量。评估最终参考时从 22 个连杆原点重编码，人类格式 6D 角是从点序列的 IK 重建。绕末端腕偏航关节自身转动时，腕点原点保持不动且下游手指无对应点，因而腕姿态的独立变化不能由这一输入直接观察。Pour 的物体倾角和闭手信号也未进入嵌入评估。脚滑判据以踝滚转连杆 5 cm 近地代理和 .5 m/s 阈值计算，0% 表示该代理通过。已有项目四项执行成功均 0/100，采用原 Pick 场景参数的补充运行仍为 0/100、右手接触 0/100。证据为 `src/g1_fk.py:13–20`、`src/feature263.py:39–65`、`src/evaluate_generated_reference_embeddings_20261105.py:162–173`、`src/evaluate_generated_reference_geometry_20261105.py:79–91` 和 `reports/DREAMCONTROL_UPSTREAM_EVAL_ALIGNMENT_20261105.md:3–7`。

## 后续修正顺序

1. 先统一世界、训练 canonical、后优化之间的高度基准，并把同一变换和逆变换随每个样本存档。用静止模板验收三处所用坐标。
2. 在后续获授权的同噪声小批比较里，把高度基准、梯度引导和训练覆盖分开。分列原始腕误差、同编码 FID 和后优化变形量，保留现有原始与最终的配对基准。
3. 分开报告论文 Full-Mix 对照、项目留出对照、筛选后的几何参考和物体执行。后优化若需要降低动作改写，应同时记录对原点列的偏差、手腕目标、真实掌指接触与物体状态，不能只用 FID 或腕点约束筛选。


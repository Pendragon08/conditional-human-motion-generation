# Conditional Human Motion Generation

**2026-10-12 C1完整对比已启动，结果尚未完成。** V28独立文档见证的60项检查通过，72条校验轨迹、传感器、策略动作与V24逐字节相同，评分一致。当前运行使用原始固定目标、3生成种子、48个测试来源、3种目标偏移，10个对比与消融配置各432输入。首次启动已确认实际父进程及采集进程存活，进度文件只作采集快照；不以启动或校验通过声称创新有效。原有预算、失败记录和后续阶段门槛保留。[独立见证](results/condition_feedback_formal_v28_20261012/DOCUMENT_WITNESS.md) · [协议与复现入口](docs/condition_feedback_formal_v28_20261012.md)。

**2026-10-12 当前状态：3个候选机制，1个初测、2个未测试，0个确认创新。** 重置、坐标、参考时钟、质心速度、已知静止/滑动/腾空对照和数值复算已校验。修正后的V24验证完成432次尝试：419次完整执行、114次条件合格、318次跟踪合格，物理与联合合格均为1次。V27固定参考回放各72次：原生成参考条件合格23、跟踪合格54；物理轨迹参考条件合格14、跟踪合格68；两者联合均为0。这些是验证集基线诊断，不是创新有效性结果。历史未修正评分与所有失败保留，不与新评分混用。原始目标、判据、冻结生成器和跟踪器不变。足部滑移仍为经过校准的表面速度代理；不作为所有滚动接触或真实接触力的真值。G1和下游训练尚未开始，本轮不写Overleaf结果。[数值环境修复见证](results/contact_com_runtime_v26_20261012/DOCUMENT_WITNESS.md) · [432次验证](results/contact_com_validation_v24_20261012/validation_summary.json) · [固定参考回放](results/fixed_reference_com_v27_20261012/replay_summary.json) · [最新HTML](results/contact_measurement_review_20261011/measurement_review.html)。

**2026-10-11 当前验证进展：3个候选机制，1个初测、2个未测试，0个确认创新。** 新增两轮人体完成等待复核和被动足部传感器复核：运动学及策略输入重复一致，净接触力读数仍不稳定；足部约束传感器可重复，但包含内部关节反力，尚未替换接触指标。5kg物体/平面校准两次通过，仍不等于人体测量门通过。完整失败与独立审计保留，继续测量修复；正式创新验证、G1和下游训练由原方案门控，本轮不写Overleaf实验结果。[最新HTML](results/contact_measurement_review_20261011/measurement_review.html) · [结果状态](results/contact_measurement_review_20261011/status.json) · [候选机制与最近邻方法](docs/candidate_mechanisms_20261011.md) · [V8独立审计](results/contact_sampling_v8_20261011/EXPERIMENT_AUDIT.md)。

**2026-10-11 前一阶段：接触测量门未通过。** 216次导出重置缓存/坐标检查完成，初始导出状态一致；四臂等预算诊断完成108次构建＋48次选出参考回放。反馈组目标误差10.80cm、随机筛选14.35cm，但物理/联合合格均为0。额外24次同步采样复核仍出现相同运动学、不同接触力的现象，**不能确认创新有效性或完整测量已修复**。仅4个自定义诊断来源/1种子，不作官方独立效能结论。保留全部失败；正式测试、G1和下游训练暂停，本轮不写Overleaf。见[最新结果表](results/condition_feedback_probe_v2_20261011/RESULTS.md)、[216次校验及边界](results/reset_alignment_v4_20261011/RESULTS.md)、[接触复核](results/contact_sampling_v5_20261011/RESULTS.md)和[HTML报告](results/reset_alignment_v4_20261011/measurement_report.html)。

**2026-10-11 实验结果入口**：[GitHub 可直接阅读的结果汇总表](results/physics_guided_20261011/RESULTS.md) · [公开结果文件与散列索引](results/physics_guided_20261011/artifact_index.csv)。完整数值、消融、失败与审计均已归档；本轮因测量校验未通过停止，G1 和下游训练未执行。

**2026-10-11：物理引导条件反馈诊断已归档，测量门未通过。** 冻结DiP和CLoSD跟踪器，实现10个对比/消融配置，各432次独立回放。独立审计发现创建锚点与实际重置坐标错位，旧轮不具备受控效力。三次独立初始化复核保留失败：V3计划216例、实际12例；首批XY对齐改善，但4项导出刚体状态不一致，随后资源保护断言停止。按方案未开展G1或下游训练，不能宣称候选机制有效或无效。保守累计约1.989 GPU小时，无旧实验删除。[HTML报告](results/physics_guided_20261011/experiment_report.html) · [旧轮独立审计](results/physics_guided_20261011/EXPERIMENT_AUDIT.md) · [V3复核与停止依据](results/reset_alignment_v3_20261011/EXPERIMENT_AUDIT.md)。

**最新汇报归档（2026-10-02）**：[HTML报告与本地视频/曲线](reports/teacher_report_20261002/README.md) · [报告入口](reports/teacher_report_20261002/实验结果汇报.html) · [DreamControl独立报告](reports/teacher_report_20261002/DreamControl_实验结果汇报.html) · [CLoSD独立报告](reports/teacher_report_20261002/CLoSD_实验结果汇报.html) · [PPO曲线解读和策略文件格式](reports/16_PPO_training_interpretation_and_policy_formats_20261002.md)。下载仓库后用浏览器打开HTML；GitHub文件页本身不会播放整个报告。正文保持论文基线汇报范围，原始许可资产和模型权重不随此报告包上传。

当前已归档DreamControl控制任务为 **11/11**：Kick85.3%、Punch87.2%、Jump100%、Sit100%、Pick93.9%、Bimanual-Pick99.9%、Ground-Pick38.1%、Ground-Pick-Top0%、Button-Press99.9%、Open-Drawer100%、Open-Door99.0%。这是保存的评估结果，不是服务器实时状态或整篇论文完成比例；Ground-Pick与Ground-Pick-Top尚无MuJoCo验证；0%保留为正式失败成绩，不代表效果复现通过。历史阶段快照在下文保留，任务时间线以[多任务基线报告](reports/03_dreamcontrol_multitask_baselines.md)为准。

2026-10-06用户重新授权研究两项候选：CLoSD式执行反馈F与短时世界模型W。首轮完成：9条人体生成接口对照、1,096回合动力学采集、四神经预测臂×三种子及440回合MuJoCo真实控制对照。F改变生成但尚无在线/执行收益；W的H10预测门和局部仿真器oracle门通过，**学习模型真实控制门FAIL**（恢复AUC点改善15.21%但CI跨零、仅比平滑低1.07%、跌倒40%→42.5%）。不启动F×W或正式PPO，完整创新未验证。已有基线和教师汇报保持原口径。[顶刊/顶会调研](reports/20_execution_feedback_world_model_literature_20261006.md) · [首轮可行性结果与负结果](reports/21_feedback_world_model_feasibility_20261006.md) · [固定消融计划](plans/feedback_world_model_ablation_20261006.md)。不使用原始重定向后的额外机器人IK，不把旧I1结果当本轮验证。

本仓库记录**条件驱动人体运动生成**课题的可复现实验。研究流程是条件输入 → 生成人体运动 → 物理／接触修正 → 机器人参考动作 → 固定下游执行与反馈。机器人跟踪用于检验生成动作是否可执行，并非本课题的唯一研究对象。本仓库与原人形机器人格斗项目分开。

2026-10-02 04:55 UTC（北京时间10月2日12:55）：用户决定暂不继续创新点验证，先完成 DreamControl 与 CLoSD 原论文复现。DreamControl已完成 **6/11** 个控制任务：Kick85.3%、Punch87.2%、Jump100.0%、Sit100.0%、Pick93.9%、Bimanual-Pick99.9%（Sit与Bimanual-Pick使用官方USD I/O显示适配，失败记录保留）；Ground-Pick已自动开始，后续四项排队。CLoSD两组官方已发布权重的4096环境评估已正常完成，新的3072环境4000轮微调仍等待当前实际场景的本地可视确认。完整验收边界见[基线复现进度](reports/03_baseline_summary.md)，已完成控制器的[结果视频](media/DreamControl_official_baselines.html)。

同日新增两项冻结策略的 **Isaac Lab → MuJoCo 首轮 sim-to-sim**：Kick 到达虚拟目标 54/66（81.8%），到达且完整 10 秒不摔倒 48/66（72.7%）；Punch 对应 42/100（42.0%）、33/100（33.0%）。无重新训练或额外 IK，两个策略的观测/动作与原检查点已核对。每条参考单独测试，口径不同于官方自动重置评估，尚未通过扰动鲁棒性或实机验收。见[报告](reports/10_Kick_Punch_MuJoCo_sim2sim_20261001.md)与[真实 MuJoCo 视频](media/sim2sim_mujoco/results.html)。

随后完成 **2,324 次冻结策略扰动／实体接触测试**：覆盖实际接触摩擦 0.5/1.0/1.5、质量 ±10%、力矩上限 −10%、左右 20/40 N 推力及固定打击靶。原位置靶的合格打击率 Kick 39/66（59.1%）、Punch 42/100（42.0%）；打击且完整 10 秒站稳分别仅 9/66（13.6%）、33/100（33.0%）。高摩擦条件下两项明显退化，不能宣称鲁棒性通过。首轮地面摩擦合并配置问题已留档并修正后完整重跑；最终 V2 有逐接触摩擦核验。[完整报告](reports/11_Kick_Punch_robustness_contact_20261001.md)与[12 段场景视频](media/mujoco_robust_contact/results.html)。这不增加论文训练任务完成数，也不代表实机打击或创新点验证。

## 当前可核查结果

| 实验 | 数据与预算 | 评估口径 | 当前结果 |
| --- | --- | --- | --- |
| DreamControl 官方 Kick 基线 | 官方 66 条轨迹，8192 环境，2000 轮，seed 42 | 官方 `play_eval.py`，1000 环境、500 帧 | **85.3%** |
| DreamControl 官方 Punch 基线 | 官方100条轨迹，8192环境，2000轮，seed42 | 官方1000环境、500帧 | **87.2%** |
| DreamControl 官方 Jump 基线 | 官方1条参考，8192环境，2000轮，seed42 | 官方1000环境、500帧 | **100.0%**，模型/视频哈希已核验 |
| DreamControl 官方 Sit 基线 | 官方1条参考，8192环境，2000轮，seed42 | 官方1000环境500帧，--disable_fabric显示适配 | **100.0%**；仅官方目标判据，不等于自然坐姿验收 |
| DreamControl 官方 Pick 基线 | 官方67条参考，8192环境，2000轮，seed42 | 官方1000环境500帧，默认显示路径 | **93.9%**，正常退出，模型/视频已核验 |
| DreamControl 官方 Bimanual-Pick 基线 | 官方1条参考，8192环境，2000轮，seed42 | 官方1000环境500帧，USD显示适配 | **99.9%**，正常退出，视频/模型已核验 |
| DreamControl 官方 Ground-Pick 基线 | 官方1条参考，8192环境，2000轮，seed42 | 官方1000环境500帧，USD显示适配 | **38.1%**，正常退出，视频/模型已核验；未做MuJoCo验证 |
| CLoSD 官方已发布权重评估 | 4096环境、500帧、CFG7.5、seed0，每状态至少1000次重置 | 未微调与已发布微调模型同口径 | 触达100%→100%、共用STRIKE85%→94%、坐下76%→89%、起身56%→97%；不是我们新微调成绩 |
| 旧 66 条策略在新动作留出集 | 1024 环境训练；32 条 seed42 留出轨迹 | 1000 环境、500 帧 | **0%** |
| 98 条增强组在同一留出集 | 原 66＋新增 32 条，1024 环境，seed 123 | 同一 32 条留出、1000 环境、500 帧 | **0%** |
| CLoSD 多任务闭环微调 | 官方预训练控制器，512 环境，增加 4000 epoch 后保存检查点 | 同口径微调前／后评估 | REACH 100%→0%、共用 STRIKE 87%→5%、SIT 77%→7%、GET_UP 56%→69%；训练在保存后异常退出 |
| I1 脚部约束后处理诊断 | 同一组 100 个 Kick 约束，生成种子 42/43/44；仅修改成功重定向轨迹的右腿 | 外部目标几何误差、固定策略仿真、接触与耗时 | 几何误差下降，但修正后跟踪显著退化；**完整 I1 Gate 未通过** |

**85.3% 与两个 0% 不能直接比较数据增益**：训练环境数、种子和测试动作不同。增强组的同口径留出集实验尚未显示成功率收益。CLoSD 512 环境缩容微调并非论文 4096 环境的严格同配置复现；Kick 与 I1 的单动作类别结果也不能代表通用生成。准确的时间线、失败尝试与限制见 [DreamControl 记录](reports/02_dreamcontrol_reproduction.md)、[CLoSD 记录](reports/01_closd_reproduction.md)及 [I1 执行与质量诊断](reports/05_I1_dynamics_contact_semantics_efficiency.md)。

## 内容

- `plans/R3_experiment_plan.md`：当前通用条件动作生成研究方案与阶段门。
- `configs/`：项目本地的版本化环境规格；上游 Isaac Sim、Isaac Gym、DreamControl 和 CLoSD 均由各自官方来源获取。
- `scripts/`：复现、资产核验、条件动作生成、质量审计、训练／评估与监控脚本。
- `results/`：机器可读的实验口径与结果索引。
- `reports/04_I1_feasibility.md`、`reports/05_I1_dynamics_contact_semantics_efficiency.md`：空间条件、机器人运动学、固定策略仿真、接触代理、语义代理及生成耗时的阶段证据；均未宣称完整 I1 成立。
- `reports/03_dreamcontrol_multitask_baselines.md`：多任务参考数据、官方预算复现口径与当前阶段状态。
- `media/`：已完成评估的小型视频。
- `checkpoints/`：DreamControl 实验的控制策略检查点，使用 Git LFS；**不含**受单独许可约束的 CLoSD、SMPL、AMASS 或 HumanML3D 资产。

DreamControl 官方代码来源记录为提交 `ec28b508313d6837de52c68247fbc8cf81fe4935`，但当前服务器的 `third_party/DreamControl` 目录不是独立 Git checkout，不能仅凭该目录再次证明完整代码版本；本轮关键源文件哈希见 `results/r3_i1_provenance_manifest_v2.json`。CLoSD 官方代码来源记录为提交 `d8950bdb78312fcd2844f336b3fc079e7368c87f`。新任务的服务器工作目录为 `/data/code/ouyanghao/oyh/conditional-human-motion-generation`，DreamControl 源码、参考动作和后续日志独立保存；通用 Isaac Sim 安装与 Python 依赖仍作为只读共享工具使用。此前运行的 CLoSD 训练保持原路径 `/data/code/ouyanghao/oyh/humanoid-combat-motion` 直到完成，避免移动文件中断进程；其结果再归档到本仓库。

官方 DreamControl 的论文报告11种任务。公开参考轨迹已逐文件校验，见 `results/official_other_tasks_assets_manifest.json` 与 `results/r3_dreamcontrol_remaining_manifest.json`；输入就绪不表示训练完成。Punch在旧图形耦合运行中第1618轮停滞，后来续训完成87.2%；新的独立训练/监控九任务队列已正常完成Jump并自动进入Sit；Sit视频评估经官方USD I/O运行适配恢复后，自动接续Pick；原失败记录保留。不得把六项成绩推广到其余任务，Jump单参考下的100%也不证明未见条件泛化。[论文与任务表](https://genrobo.github.io/DreamControl/static/paper/DreamControl_arxiv.pdf)。

## 数据与许可

本仓库不含官方大型动作库、模型权重或机器人资产。CLoSD 使用的 SMPL、AMASS 与 HumanML3D 需遵守各自许可；因此其原始数据及由其衍生的控制器权重不会未经单独核实直接发布。DreamControl 的官方资产和生成轨迹请按原项目说明获取并校验哈希。完整训练日志与较大的中间产物保留在服务器，`results/` 记录必要的路径、参数和摘要。

2026-10-02补做四项已完成控制器的 **MuJoCo冻结策略迁移**：Jump、Sit、Pick、Bimanual-Pick共70条参考，均正常结束测试并保存5段视频。Jump摔倒，Sit达参考目标但未持续接触座椅，两种拾取未抬升物体；不能宣称四项迁移通过。此口径不等于Isaac的1000环境自动重置成功率，也没有新增训练。[视频与逐项结果](media/sim2sim_completed_20261002/results.html) · [接口、场景和接触核验报告](reports/14_Completed_tasks_MuJoCo_sim2sim_20261002.md)。截至此轮，已完成的六项控制任务都有跨仿真测试记录；可靠执行程度需按各自指标判断。

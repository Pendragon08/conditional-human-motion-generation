## Latest user direction: evaluate at10k and improve semantics

User2026-10-26 requests physical filtering assessment at10,000 teacher rounds
before deciding whether40,000 is needed. Originalpaper baseline remains frozen.
See [10k evaluation and semantic plan](reports/teacher10000_and_semantics_20261026.md)
and [runtime contract](TEACHER10000_EVALUATION_ENVIRONMENT.md).
Quick online functional chain is accepted; semantic/paper quality is still not.
New lexical coverage audit found only1wave-stem annotation (~0.84s) inquickTRAIN,
versus147availableTRAINsource candidates (27explicitwave+right); this is a lead
for realphysics and controlled semantic tests, not a causal or quality claim.
Separate [post-hoc 5k endpoint baseline](reports/recorded_endpoint_baseline_20261027.md)
compares retained terminal poses; it adds no physics trial or acceptance threshold.
Future fullpaper completion must include accepteddata/fullgenerator/requiredWBC,
augmentation/onlinequality/video; user must be notified when actually attained.

# TextOp 复现

当前已验证**快速版在线“文本切换→异步生成→真实仿真反馈控制”功能链路**：作者及独立两次各257步均零超时、断流和物理失败。语义与论文质量仍未通过；跟踪器已从5k状态实际恢复，先到10k备份并受控暂停，比较筛选效果后决定是否继续40k。实际状态见[在线验收与续训报告](reports/online_quick_20261026.md)。最新证据与本地录像入口见[实际结果报告](reports/quick_actual_results_20261026.md)，继续工作先读[CONTINUATION顶部](CONTINUATION.md)。

|环节|已核验状态|
|---|---|
|可得数据预处理|8278原始来源及镜像完成GMR、文本对齐、69D/FK与私有备份|
|快速训练语料|64条实测中42自然完成、22失败；42原始参考保留31train/11val、174标注对、74非空CLIP文本|
|快速训练|5k跟踪策略、VAE2k、LDM3k均完成；完整状态在服务器及D盘私有核验备份|
|实际生成|真实训练权重、history2→future8、5步CFG5、32段/256未来帧；两次独立生成数组一致|
|生成参考的物理执行|同一258姿态序列作者/独立两次均自然完成257控制步/1028受控物理子步，文本切换目标130已消费|
|新执行录像|258帧、50fps，实际执行姿态可视化；本地私有视频与独立审计完成|
|在线功能验证|脚本实时文本切换，异步生成和实际50Hz控制；两次各257步零超时/断流/物理失败，独立63+430+41检查通过|
|完整训练接续|已从已核验5k完整状态实际恢复，先在10k评估筛选效果；沿用原8192环境GPU物理和保留的40k配置；完整VAE/LDM仍等待全量物理筛选与最终语料|
|尚未验收|语义/论文质量、无限时长实时性、最终端点、论文最终WBC与完整训练结果|

固定完整采样Huber由0.310267升到0.607536，生成质量仍待评估；短序列被跟踪不等于语义或论文质量达标。自定义身体平均误差0.043041m、关节RMSE0.128405rad只描述本次控制目标跟踪。原生SDK关闭仍使用有界兜底，差异保留。作者精确碰撞资产不可得。

快速功能链路通过后，按用户最新要求先用10k跟踪器评估筛选效果；根据实际结果扩大筛选或继续训练，并保留40k论文参照。完成全部可得数据真实筛选、最终语料及容量验收后再进入论文VAE200k/LDM300k；快版课程/学习率时钟已缩短，不把快版计数当作论文完整结果。受限动作/逐条文本/训练全状态保持私有，公开仓库保存代码、聚合证据、报告和许可允许的视频。

新配置与独立证据：[生成入口](QUICK_GENERATION_ENVIRONMENT.md)、[SDK生成执行](QUICK_GENERATED_EXECUTION_ENVIRONMENT.md)、[实际执行录像](GENERATED_EXECUTION_VIDEO_ENVIRONMENT.md)、[独立SDK审计](results/generated_execution_20261026/independent_document_v1/REPORT.md)、[独立媒体审计](results/generated_execution_20261026/independent_video_document_v1/REPORT.md)。历史SFU参考跟踪科研视频已[公开发布](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/textop-quick-tracking-video-20261026)；新生成执行视频暂存本地私有。

<details>
<summary>保留的历史阶段记录（不作为当前运行指令）</summary>

以下“当前/最新/等待/禁用”等属于历史记录；实际状态以CONTINUATION最顶部和冻结运行的真实收据为准。不要据此重启已结束工作者或重复已完成的诊断。


[真实评估到quick语料的来源选择入口](reports/quick_source_selection_20261026.md)：已部署禁用候选，本地/服务器15项合成逻辑检查通过；真实64来源决定、可训练语料和生成器仍未完成。

[快速生成器方案与两卷预算](reports/quick_generator_proposal_20261026.md)：准备了尚未启用的VAE2,000/LDM3,000小样本方案及保留检查点的存储安排。原200k/300k配置未改，真实保留数据和该方案训练入口仍待验收；不是训练启动或论文完成。

[执行轨迹视频渲染准备](reports/execution_video_20261026.md)：已通过原服务器CPU/Mesa作者及fresh独立渲染检查，旧地面闪烁及中间错误计数保留并修正。实际策略视频仍为0，teacher继续到5,000轮；渲染不替代物理试验或生成闭环。

[快速5k回放候选与实际选择范围](reports/quick_pilot_candidate_20261026.md)：独立禁用入口已部署，本地/服务器31项CPU检查及fresh文档16项只读检查通过；实际SDK回放和端点核查尚未完成。当前策略继续向5,000轮节点运行，协调器所有权不变。

**2026-10-26当前优先：快速版5,000轮。** 用户已选择先加快复现；自动切换协调器已启用，等待第5,000轮完整检查点稳定和备份后受控结束，再做小规模真实跟踪评估。原40,000轮配置与完整版验收口径保留，当前不算论文同规模结果。见[快速复现安排](reports/quick_reproduction_20261026.md)与CONTINUATION顶部。以下40,000轮启动记录属于保留的原方案历史。

**2026-10-26最新：服务器已启动 40,000 轮特权跟踪策略正式训练。** [地面接触修复、独立验证与正式启动报告](reports/contact_isolation_20261026.md)：两次完整参考库短 PPO 验证通过后，正式运行已取得 1,600 轮 / 32,000 次实际优化更新的证据。此前 8,278 源动作预处理完成；真实两轮物理筛选、完整 VAE/LDM 和生成—执行闭环尚未完成。原 Plane、自碰撞对照、所有失败和私有数据均保留。当前活动入口和禁止重复启动规则见 CONTINUATION 顶部。以下旧记录是历史状态。

[训练与筛选接触配置一致性及退出后验收准备](reports/filter_contact_protocol_20261026.md)：本地和服务器各 23 项合成逻辑检查通过；新回放器仍禁用，尚未进行真实 SDK 验证，不代表物理筛选完成。

[后续筛选资源监测与失败保留](reports/filter_supervision_20261026.md)：7项实际CPU子进程检查与更新后本地/服务器各27项验收逻辑检查通过；新SDK回放入口仍禁用，活动训练保持运行。

[首个1,000轮正式策略检查点备份](reports/formal_checkpoint_backup_20261026.md)：完整状态保存在服务器及本地私有目录，去除逐动作统计的推理权重已在GitHub核验；训练继续，尚未完成物理筛选。

[有界回放验证入口准备](reports/contact_replay_v4_20261026.md)：新增独立验证模式，本地/服务器各13项CPU检查通过；完整40,000轮策略仍是前提，三个启动/验证标志均禁用，实际SDK验证尚未执行。

23DoF/57D基线训练现场已暂停；优先准备论文29DoF/69D数据。

优先查看 [数据准备核查与重建报告](reports/dataset_preparation_20261025.md)。已获授权的数据验收后续步骤见 [Step2运动生成器训练安排](reports/step2_motion_generator_20261025.md)。另见 [启动报告](reports/启动报告_20261024.md)、[环境说明](ENVIRONMENT.md)、[官方版本](upstream/README.md)。

最新验证：[空间清理与CMU续传](reports/storage_cleanup_20261025.md)、[69D生成器核心数值验证](reports/paper69_runtime_validation_20261025.md)。完整69D生成器尚未训练；候选数据尚未通过动态物理筛选。

新增：[真实跟踪MDP、参考导出、PPO更新与断点恢复验证](reports/tracker_integration_20261025.md)。643条源动作及镜像的参考导出和私有备份已完成；1,653条新源动作也已重定向并逐条校验备份。CMU1,330条已完成重定向与仿真参考导出，持久备份状态以最新队列记录为准。BMLrub/KIT完整原始包及全部对应索引已核验，按串行队列接续。

8,192环境已完成真实容量验证，各20次更新；新增统计恢复层完成40次真实更新及模型/Adam/归一化/EMA/真实尝试计数恢复。均有独立文档复跑，不代表40,000轮训练或跟踪质量通过。正式筛选训练代码见[FILTER_TRAINING_ENVIRONMENT.md](FILTER_TRAINING_ENVIRONMENT.md)，当前被完整数据与容量验收条件禁用。后台继续准备数据和真实筛选，验收后按VAE200k→LDM300k启动完整69D生成器；旧57D训练继续暂停。持续工作入口见[CONTINUATION.md](CONTINUATION.md)。

[物理筛选验收规则v2](FILTER_EVIDENCE_PROTOCOL.md)补充实际训练凭据和全轨迹覆盖核对；17项合成逻辑测试通过，不等于已执行物理评估或数据已通过。

[严格SDK回放执行器](STRICT_REPLAY_ENVIRONMENT.md)已实现并进行诊断：完整原动作仍失败，另有明确2帧控制流夹具验证结束信号；没有原动作接纳。修复重置相对参考缓冲区导致的首步高度误判。CMU/BMLrub/KIT参考导出队列会在各批GMR完成后串行接续，正式两轮筛选仍等待真实40,000轮策略。

[候选语料加载验证](CANDIDATE_LOADER_ENVIRONMENT.md)已建立2,296条完成源的未过滤视图，检查真实CLIP缓存、训练集统计与官方batch512加载器。发现1条训练源与验证源的全长动作重复，在候选视图中隔离训练源及镜像，原始数据与验证集保留；这不是物理筛选通过。最终范围验收见[SOURCE_SCOPE_PROTOCOL.md](SOURCE_SCOPE_PROTOCOL.md)：明确13条不可获得来源，仍要求其余8,278条全部有真实准备和过滤记录。

[个人目录空间清理已完成](reports/personal_storage_cleanup_20261025.md)：完整备份并核验GitHub后清理闲置EngineAI环境和旧缓存，数据盘回收9.66GiB；系统盘19个资源文件迁到个人数据盘并保留原路径，SHA全部一致。最终采样数据盘余6.97GiB、系统盘余6.23GiB，数据/模型/实验检查点/唯一记录保留，GMR未中断。[旧EngineAI环境恢复说明](ENGINEAI_ENV_RECOVERY.md)。

[BABEL原始标注已下载、安装服务器并独立核查](reports/babel_original_alignment_20261025.md)：8,291条运动对应50,004条分段标签与原始JSON一致；原始标注及逐条索引按许可保存在个人目录。

官方代码以固定提交镜像保存于upstream/，下载数据不入Git。scripts/为本次下载、隔离环境安装、训练启动与指标记录代码。官方算法源文件未修改；论文29DoF重建部分为独立自实现并复用GMR核心，尚待真实动态筛选。

[完整网络batch512数值检查](NEURAL_BATCH512_ENVIRONMENT.md)已由作者和独立文档执行验证：8次VAE、4次LDM反向，0次优化器更新。显存测量未分配Adam/EMA状态，不代表完整训练已启动；[实际范围与报告](reports/neural_batch512_20261025.md)。

[预处理持久备份验收](PREPROCESSING_BACKUP_AUDIT.md)逐条读取已备份的动作核对SHA、数组和来源覆盖；局部批次通过不代表全部数据完成，也不替代真实物理筛选。[聚合报告](reports/preprocessing_backup_audit_20261025.md)。

[生成器优化器与断点恢复诊断](GENERATOR_STATE_ENVIRONMENT.md)补充真实Adam/AdamW状态分配、落盘和新管理器恢复。每阶段仅8次诊断更新，候选数据未通过物理筛选；[聚合结果](reports/generator_state_20261025.md)明确区分正式训练、EMA测试设置及实际检查点步数。

[GMR四进程并行已启用](PARALLEL_GMR_ENVIRONMENT.md)：仍限制CPU0–3，作者/独立完整动作对照约2.41/2.36倍加速，所有比较数组差异0。BMLrub已从完整上传数据启动四进程，KIT后续沿用；此前精度失败与互斥保护记录保留。[结果与适用范围](reports/parallel_gmr_20261025.md)。

[后续训练存储预算](reports/training_storage_20261025.md)已按现有配置和实测诊断检查点估算；当前数据盘不足以保留计划的全部训练检查点，正式训练前需落实持久存储方案。并行重定向继续，数据和检查点未删除。

[2026-10-26存储与训练预算更新](reports/training_storage_20261026.md)：闲置软件备份后回收约5.49GiB，数据和检查点保留；按正式EMA关闭设置重新估算三阶段检查点，分盘方案仍需启动前验收。

[完整参考库训练前核查](reports/full_library_capacity_20261026.md)：5,765条已完成源与仿真参考的逐条SHA、来源/划分和镜像对应核查通过；KIT仍在现有队列中。完整8,278源的8192环境容量入口已准备但禁用，尚未执行完整库GPU验证；[接续条件](FULL_LIBRARY_CAPACITY_ENVIRONMENT.md)。

[正式筛选检查点存储保护](reports/checkpoint_storage_guard_20261026.md)增加保存前的个人目录、持久盘和剩余容量检查；不把低于保留空间后再尝试保存当作可靠保护。训练启动条件保持不变。

</details>

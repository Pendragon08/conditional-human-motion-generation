# TextOp 复现

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

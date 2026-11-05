# 训练任务自动接续

用户最新明确跳过第四步Sim2Real Deployment。全部任务以Isaac Lab / Isaac Sim（PhysX）的物理仿真验证和评估为终点：保留策略检查点、测试条件、真实成功率及失败记录，不将仿真结果当成真机有效性。无需硬件实验即可完成本项目范围内的任务复现。

已按2026-11-03用户要求启用自动接续。范围是[论文Table III](https://arxiv.org/html/2604.00202#S4.T3)中的8项技能，采用本项目29DoF改编：

Pick PPO训练/评估 → SuperMimic框架接入及同一Pick任务训练/评估 → Punch → Deep Squat → Deep Squat & Pick → Drawer Open → Pour → Vertical Wipe → Step & Punch。SuperMimic阶段是用户新加的框架替换步骤，不是额外的论文技能；后续优先使用验证过的新框架。

Pick已经在训练。用户最新将当前Pick总预算改为3000轮，包含已保存的251轮。结束后，服务器控制器会核对最终检查点`model_2999.pt`，并自动进行100环境评估、归档实际成功率，然后交接下一任务。完成训练预算、完成评估、取得任务成功是三个不同结论；不能把计划轮数或loss下降当成任务成功。评估失败留证修复，不用4环境入口冒烟冒称100环境结果。其余任务的默认预算仍单独记录，当前改动只覆盖Pick。

接续自动化`dreamcontrol-v2`已确认ACTIVE，每30分钟在本聊天继续推进。其余任务的29DoF适配、场景、生成参考与验收尚需实现，也已纳入后续自动执行；不声称七个后续任务的训练代码已经全部就绪。旧v1其他项目的27DoF任务队列不启动、不改名充当新任务。

每一项依次执行：任务专用参考生成与29DoF后优化、场景及输入/动作检查、小规模真实仿真/PPO、资源复核、正式训练、100环境评估和结果归档。复用当前扩散先验与已交付数据，不重新预处理原始GRAB或增加GRAB/Omni源动作筛选。Actor继续只读实时本体和任务状态，参考用于奖励/初始化。保留所有检查点与失败记录，不做真机部署。

服务器记录目录：`/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_skill_sequence_20261103`。

- `sequence.json`：任务范围、顺序、论文依据、哪些适配仍待实现。
- `progress.json`：当前任务与进度。
- `active_controller.json`、`controller_supermimic_handoff_v1_launch.json`、`controller_supermimic_handoff_v1.log`：当前后台控制器PID1348982，不重复启动；旧controller日志保留。
- `stage3_supermimic29_20261103/handoff_plan.json`：当前评估后优先执行的框架替换计划，匹配8912环境及3000×24控制步预算；原生优化器迭代另计，不盲目复用PPO损失或把未来参考输入Actor。复用已配置TextOp环境，不改其他项目。
- Pick活动运行、目标轮数及保护器以`stage3_pick29_20261103/active_pick_run.json`为准，当前`runs/pick29_8912_3000_resume_v3`，从旧`model_250.pt`恢复，不覆盖旧结果。
- `src/stage3_skill_queue.py`：只负责Pick完成后的评估和交接；后续七任务由本聊天接续自动化实现与推进。
- `evaluations/pick_final_3000_v1/report.json`：Pick完成后实际生成的正式评估记录，目前尚未生成。
- `evaluations/pick_entrypoint_smoke_v1/report.json`：已完成的4环境4步只读模型入口冒烟，0训练更新、检查点哈希不变，不是成功率评估。
- `completions/`：各任务预算和评估归档，成功率如实记录。

后台控制器持有`controller.lock`时不要并发修改`progress.json`，可读取进度并在`prepared/`独立准备后续任务。接续自动化在当前任务训练/评估正常时保持安静；任务完成、实质失败或需要用户行动时才通知。

Pick评估100个环境的物体初始XY为当前标定桌面内的均匀±0.06m范围，成功按物体高于0.95m且不早于接触时刻统计；它是明确标记的项目基线分布，不冒称与论文完整测试分布一致。固定首次episode计数，提前结束的失败环境保留在成功率分母中。

# TextOp 复现

23DoF/57D基线训练现场已暂停；优先准备论文29DoF/69D数据。

优先查看 [数据准备核查与重建报告](reports/dataset_preparation_20261025.md)。已获授权的数据验收后续步骤见 [Step2运动生成器训练安排](reports/step2_motion_generator_20261025.md)。另见 [启动报告](reports/启动报告_20261024.md)、[环境说明](ENVIRONMENT.md)、[官方版本](upstream/README.md)。

最新验证：[空间清理与CMU续传](reports/storage_cleanup_20261025.md)、[69D生成器核心数值验证](reports/paper69_runtime_validation_20261025.md)。完整69D生成器尚未训练；候选数据尚未通过动态物理筛选。

新增：[真实跟踪MDP、参考导出、PPO更新与断点恢复验证](reports/tracker_integration_20261025.md)。643条源动作及镜像的参考导出和私有备份已完成；1,653条新源动作也已重定向并逐条校验备份，正在继续导出其仿真参考，CMU1,330条源动作接续重定向。BMLrub/KIT完整原始包及全部对应索引已核验。

8,192环境已完成真实容量验证，各20次更新；新增统计恢复层完成40次真实更新及模型/Adam/归一化/EMA/真实尝试计数恢复。均有独立文档复跑，不代表40,000轮训练或跟踪质量通过。正式筛选训练代码见[FILTER_TRAINING_ENVIRONMENT.md](FILTER_TRAINING_ENVIRONMENT.md)，当前被完整数据与容量验收条件禁用。后台继续准备数据和真实筛选，验收后按VAE200k→LDM300k启动完整69D生成器；旧57D训练继续暂停。持续工作入口见[CONTINUATION.md](CONTINUATION.md)。

[个人目录空间清理已完成](reports/personal_storage_cleanup_20261025.md)：完整备份并核验GitHub后清理闲置EngineAI环境和旧缓存，数据盘回收9.66GiB；系统盘19个资源文件迁到个人数据盘并保留原路径，SHA全部一致。最终采样数据盘余6.97GiB、系统盘余6.23GiB，数据/模型/实验检查点/唯一记录保留，GMR未中断。[旧EngineAI环境恢复说明](ENGINEAI_ENV_RECOVERY.md)。

[BABEL原始标注已下载、安装服务器并独立核查](reports/babel_original_alignment_20261025.md)：8,291条运动对应50,004条分段标签与原始JSON一致；原始标注及逐条索引按许可保存在个人目录。

官方代码以固定提交镜像保存于upstream/，下载数据不入Git。scripts/为本次下载、隔离环境安装、训练启动与指标记录代码。官方算法源文件未修改；论文29DoF重建部分为独立自实现并复用GMR核心，尚待真实动态筛选。

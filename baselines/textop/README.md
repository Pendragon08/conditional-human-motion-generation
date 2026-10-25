# TextOp 复现

23DoF/57D基线训练现场已暂停；优先准备论文29DoF/69D数据。

优先查看 [数据准备核查与重建报告](reports/dataset_preparation_20261025.md)。已获授权的数据验收后续步骤见 [Step2运动生成器训练安排](reports/step2_motion_generator_20261025.md)。另见 [启动报告](reports/启动报告_20261024.md)、[环境说明](ENVIRONMENT.md)、[官方版本](upstream/README.md)。

最新验证：[空间清理与CMU续传](reports/storage_cleanup_20261025.md)、[69D生成器核心数值验证](reports/paper69_runtime_validation_20261025.md)。完整69D生成器尚未训练；候选数据尚未通过动态物理筛选。

新增：[真实跟踪MDP、参考导出、PPO更新与断点恢复验证](reports/tracker_integration_20261025.md)。短PPO诊断已通过40次实际Adam更新及恢复检查；不代表论文40,000轮过滤训练完成。首批643条源动作及镜像正在导出完整仿真参考。后台继续补齐数据和筛选，真实验收后按VAE200k→LDM300k启动完整69D生成器，旧57D训练继续暂停。持续工作入口见[CONTINUATION.md](CONTINUATION.md)。

[BABEL原始标注已下载、安装服务器并独立核查](reports/babel_original_alignment_20261025.md)：8,291条运动对应50,004条分段标签与原始JSON一致；原始标注及逐条索引按许可保存在个人目录。

官方代码以固定提交镜像保存于upstream/，下载数据不入Git。scripts/为本次下载、隔离环境安装、训练启动与指标记录代码。官方算法源文件未修改；论文29DoF重建部分为独立自实现并复用GMR核心，尚待真实动态筛选。

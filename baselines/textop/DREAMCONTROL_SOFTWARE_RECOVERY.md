# 闲置 DreamControl 软件依赖的备份与空间回收

仅处理个人目录的 dreamcontrol-retarget 软件依赖目录。它不包含本次动作数据集、模型检查点或实验结果；当前 TextOp、GMR、IsaacSim 和 IsaacLab 依赖保留。

**PBHC training 环境保留**：检查发现它是当前 TextOp Python 的基础环境，并被暂停的旧训练进程加载，因此未纳入清理。

清理前核对当前进程映射、打开文件、环境变量，以及其他环境的解释器链接和运行时依赖。所有目标文件属于本人。

完整软件归档：3553869959 字节，逐文件校验 13817 个文件；归档 SHA256：`28e157c87990d38580594b9c05f7f571d2c42f122ff1b6f99d0c82b8f4ad1a63`。

[完整恢复包](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/inactive-dreamcontrol-overlay-20261026) 包含全部有序分片、逐文件清单、软件版本和许可证。只有全部 GitHub 分片摘要、大小及代码备份核验通过，才允许清理原软件副本。

恢复：下载并按编号拼接 idle_dreamcontrol_overlay.tar.gz.part001 起的全部分片，核对各分片和完整包 SHA256，再解压到服务器原路径。原路径若已存在新环境，不得覆盖。恢复后需重新验证对应软件任务，不能仅凭解压成功称环境可用。

当前仅完成本地归档内容验证，原软件目录仍保留。最终删除状态以 cleanup.json 为准。

# 数据盘与系统盘清理、复现暂停记录（2026-10-24）

**按用户最新要求暂停现有复现及自动推进，完成两块盘的定点清理：数据盘释放4.78GiB，系统盘释放1.97GiB，共6.75GiB。数据集和最新13,001次检查点保留。**

以下余量为UTC `2026-10-24T14:45:00.040951+00:00`（北京时间2026-10-24 22:45）的实际采样；释放量按已删除文件实际占用块统计，空闲量还受其他后台活动影响。

| 分区 | 清理前可用 | 清理后可用 | 本轮释放 |
|---|---:|---:|---:|
| 数据盘 `/data` | 2.18GiB | 6.96GiB | 4.78GiB |
| 系统盘，操作限本人 `/home/ouyanghao` | 1.56GiB | 3.52GiB | 1.97GiB |

## 暂停与清理范围

`phygile-reactivebfm`自动推进已设为`PAUSED`，保持原计划和提示，未经用户恢复不会继续。ReactiveBFM训练进程此前已自行退出，本轮没有恢复训练。关闭了旧PhyGile预览HTTP服务、旧Isaac动作查看器、其自动重启监督器、遗留Sit评估及8个本人tmux监控/看板会话。Sit评估未响应正常退出信号，核对PID、UID及确切任务后强制结束；日志和模型保留。复核相关目录下的活动进程为0。未停止其他用户进程。

只删除以下已核验资产的服务器副本，共25个普通文件；没有递归清空项目目录。

1. 数据盘：v4历史10,061、11,001、12,001检查点的model/EMA文件及优化器，共6件、约4.78GiB。
2. 系统盘：旧`training_v1`的10,001次model/EMA文件及优化器，共2件、约1.59GiB。
3. 系统盘：本人Omniverse目录下17件内置Wireframe纹理缓存，共约0.37GiB。

每件历史检查点删除前都先计算服务器完整SHA256，核对GitHub Release的大小/摘要，并复核已有本地恢复副本完整SHA256。8件均匹配。完整下载链接、原路径、字节数与哈希见`results/pause_cleanup_20261024/backup_restore_index.json`；大文件无需再次重复上传。备份分别位于[论文复现结果Release](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/paper-reproduction-progress-20261023)及[旧完整训练Release](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/reactivebfm-full-training-20261022)。

17件纹理与既有已上传缓存归档内的内容逐字节哈希一致。本轮重新核对本地归档整文件SHA256，并从压缩包流式读取对应成员再次计算SHA256，全部匹配后才清理。归档为`inactive_builtin_cache_backup_v1.tar.gz`，SHA256`87fdf8cb00cc6d35b02fb2a6d7bd855d927ca19082ddc148560eac03e73fe254`，保存在[缓存备份Release](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/server-unused-cache-backup-20261023)及既有本地备份。

## 数据集与可恢复状态

4组数据集、共**166,311个文件**，清理前后文件数、逻辑总字节数和占用块数均一致。这是文件清单统计复核，没有冒称对全部数据重新进行内容哈希。HDM05原始压缩包仍保留；AMASS/SMPL资产和HuggingFace目录内的数据集文件均未选入删除白名单。生成的训练数据也没有被当作普通缓存批量清空。

最新完整13,001次模型/EMA和优化器仍在服务器，并于清理后重新计算完整SHA256：

| 保留文件 | SHA256 |
|---|---|
| model000013000.pt | `4bc375b0b1fe0d1ab7e076e3197ed1ea4d48e038b0764ae68d58b51882f9a56e` |
| opt000013000.pt | `3c3701c660fd9f4beeae3857a7fc055689612b81abf9999ff620f8c9ade12cf6` |

代码、原始运行日志、评价轨迹、视频和失败记录均保留。旧ready清单作为历史记录保留；其中已归档的旧文件不能再视为服务器本地可恢复资产。当前服务器完整恢复点为13,001次。未经GitHub备份核验的其他模型未删除。

Isaac Sim/Isaac Lab安装和现有Python环境保留。其中已有Python3.8.20环境，后续TextOp可能复用。没有通过清理系统共享目录、修改磁盘保留比例或移除其他用户文件来释放空间。

## 保护检查与失败留档

清理器只接受确切的文件白名单，检查UID1013、普通文件、单硬链接、无符号链接及祖先链接、inode/mtime/长度、内容哈希和活动占用。先预演，再执行；实际源代码快照随结果提交。

缓存预演先后两次拒绝执行，因为旧Isaac查看器仍在运行，且首次关闭后被监督器自动拉起。两次拒绝和完整错误保留。停止监督器、残留评估与监控后再次预演通过，随后才删除17件缓存。没有跳过占用检查或放宽数据保护。

## SSH与后续工作

此前服务器连接已修复：关闭SakuraCat1.6.2的TUN，保留系统代理，并为原服务器增加仅该Host生效的ZeroTier源地址绑定；3次重新认证登录均成功，每次约1秒。HTTPS代理测试HTTP200。SSH原配置已另存本地备份，服务器SSH服务及网络配置未修改。详见[连接修复记录](ssh_zerotier_repair_20261024.md)。

按用户最后的优先级，本轮仅清理和归档，没有启动TextOp下载、训练或评价，也没有新增结果视频。TextOp仍是后续目标，需核对官方所需数据与预训练模型，缺失项再从官方指定来源获取；现有PhyGile与ReactiveBFM保持暂停。

本地报告：`D:/具身智能/PhyGile与ReactiveBFM_论文对齐复现_20261022/服务器磁盘清理与暂停记录_20261024.html`，Markdown同名；本轮完整原始回执在`pause_cleanup_textop_20261024/`。公开清理代码与核验元数据在仓库`baselines/paper_faithful_reproduction/results/pause_cleanup_20261024/`，不包含受限原始数据或账户凭据。

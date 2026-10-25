# 服务器个人目录空间清理核验

完成核查：2026-10-25T14:22:13.163335+08:00（北京时间）。

| 磁盘 | 清理前可用 | 清理及资源迁移后可用 |
|---|---:|---:|
| 数据盘 | 2.42GiB | 6.97GiB |
| 系统盘 | 1.11GiB | 6.23GiB |

清理闲置EngineAI Python环境与指定旧编译、着色器、pip和安装包缓存，数据盘实际回收约9.66GiB。82,327个归档源文件逐个SHA256核对，完整压缩归档验证、全部六个GitHub分片摘要核验、代码/清单备份完成后才删除原副本。[完整恢复备份](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/inactive-engineai-environment-cache-20261025)，恢复说明见[ENGINEAI_ENV_RECOVERY.md](../ENGINEAI_ENV_RECOVERY.md)。旧EngineAI环境已归档，后续使用需先恢复；其框架源码、基础Python、当前TextOp/GMR环境及共享IsaacSim平台保留。

系统盘的TextOp资源和PhyGile保留资源，共19个文件、约5.11GiB，已移到个人数据盘retained-system-assets-20261025。逐文件SHA256一致，原/home路径保留为链接；额外通过原路径重读19个文件再次验证SHA。**数据集、人体模型、预训练模型、实验检查点和唯一实验记录的内容均保留。**迁移占用一部分数据盘空间，不能把迁移量算成两盘整体新增空间。

GMR当前批次完成1615/1653，失败0，状态RUNNING；清理没有停止或重启训练/预处理。旧57D训练保持原暂停状态。其他用户目录未修改。

约17GiB的CLoSD下载目录承载数据和预训练权重，因此保留。另在本地核验完整BMLrub原始包与前缀并备份恢复证据后，仅删除旧断点副本，D盘回收3.31GiB。

备份上传曾出现连接重置及API暂时失败；全部失败记录保留，原文件一直保留到备份核验通过。后续改由服务器直接向官方GitHub上传剩余分片；认证信息只经SSH标准输入与内存传递，未保存到代码、日志或报告。首次SSH执行阶段在未删文件时断连，经检查无执行进程、源清单零缺失后改用监督进程；清理和迁移均实测退出0。

全部清理限定在ouyanghao个人目录。受限AMASS/SMPL/BABEL及重定向资产保持私有，未公开上传。清理、迁移、GitHub校验和运行状态证据见results/storage_cleanup_20261025。数据准备与完整生成器训练仍按原验收顺序继续。

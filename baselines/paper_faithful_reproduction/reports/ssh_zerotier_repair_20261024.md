# SSH / ZeroTier连接修复（2026-10-24）

本机SakuraCat1.6.2开启TUN时，原服务器SSH超时或在握手前中断。Clash Verge的旧配置并不是当前活动代理。关闭SakuraCat TUN、保留127.0.0.1:7897系统代理，并在本机SSH配置为原服务器单独绑定ZeroTier源地址后，连续3次重新认证登录nlp03/UID1013成功，耗时约0.97、1.00、1.08秒；HTTPS代理访问GitHub API返回200。

修复支持本机TUN/出口选择存在影响，未将其等同于服务器SSH服务故障，也未量化每项设置的单独贡献。ZeroTier节点通信需要正常UDP路径，见[官方路由说明](https://docs.zerotier.com/routertips/)。没有改防火墙、服务器SSH配置或重启服务器。

只在本机`C:/Users/Pende/.ssh/config`追加原服务器Host块，保留全部已有Host。原文件备份在同目录`config.backup_20261024_zerotier_repair`。新增绑定随本机ZeroTier地址变化需要调整；恢复原设置可使用该备份。SakuraCat TUN当前关闭，原系统代理保留。账户凭据未复制到报告或GitHub。

连接恢复后发现ReactiveBFM在14,001次有效更新处等待保存空间，另一rank随后NCCL ALLREDUCE超时；监督器于UTC2026-10-23 21:25:12（北京时间10月24日05:25）退出1。最新完整磁盘检查点仍是13,001次，不能把未保存的14,001状态当作恢复点。原始退出日志已保存在本地`ssh_zerotier_repair_20261024/training_exit_and_checkpoint_check.json`。

用户随后要求暂停现有复现并先清理磁盘，因此当前不恢复训练；清理和保存范围见[暂停与磁盘清理记录](pause_cleanup_20261024.md)。本地设置截图在`ssh_zerotier_repair_20261024/SakuraCat_TUN_off.png`。

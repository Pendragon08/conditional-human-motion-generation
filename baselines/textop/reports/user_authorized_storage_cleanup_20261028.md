# 用户授权空间清理：实际完成与恢复位置

用户回复“你帮我清理”，本次新指定操作已实际退出0，数据、检查点、完整归档及唯一实验记录保留。

| 项目 | 实际结果 |
|---|---:|
| D盘四个重复运输分片回收 | 1925545992字节（约1.79GiB） |
| 服务器旧安装包私有转移 | 201141673字节（约191.8MiB） |
| D盘扣除私有安装包后的实测净增 | 1724399616字节（约1.61GiB） |
| data当前空闲 / 导出及LDM保守预算 | 4377014272 / 4320133120字节 |
| D当前空闲 / 其余三份VAE状态加2GiB保留 | 3933577216 / 3808428032字节 |

上述当前预算均通过，无需继续等待用户提供额外空间；并非磁盘预留或最终LDM接受。开训前和每次保存仍重新检查，不放宽2GiB保留线。VAE持久HOME剩余四槽预算4898947072字节也通过；WBC、未来媒体和完整LDM私有备份容量仍按实际需要单独验收。

D盘仅删除hab-contact-aware-world-model/release/contact-v1.tar.gz.part000至part003。保留完整contact-v1.tar.gz，SHA8ef8118835c1b58b8a4475b2ee9322d8584cfb140ebbf3de5e75cc85d191f14c。完整包2404个成员逐个SHA通过，四片SHA与完整包偏移范围逐字节相同，范围重新生成到哈希计数器验证通过，现有私有GitHub Release四片摘要也一致。完整包、清单、模型和原始结果保留，恢复工具provider_tools/restore_contact_transport_part_20261028.py可按清单恢复任意一片。旧断点包和失败partial未清理。

服务器仅移除本人cache/downloads中的旧IsaacGym_Preview_4.tar.gz安装包副本，201141673字节，SHAc51d77ac1144b91b356f439c7313d8d59468147e37187ebffe7cdbc3154fdb8f。完整私有副本保留在D:/具身智能/TextOp复现_20261024/software_recovery_private_20261028/IsaacGym_Preview_4.tar.gz，真实SCPexit0、所有权/字节/SHA和完整gzip CRC核验通过。安装包移动本身不算整体新增空间；运行中的已解压SDK和依赖、活动env、保护的.venv-pbhc-training均未修改。NVIDIA安装包没有公开上传。

恢复工具、精确目标和来源证明在删除前已实核GitHub提交2ea1c63b096253a6f29899f3f8aa592dc9894d7d。Windows原生PowerShell逐个固定绝对路径执行非递归删除，真实exit0、逐文件日志及前后容量保留。服务器独占意图/结果与真实SSHexit0保留。最初全局句柄检查拒绝于修改前，后续识别不可读者为SSH/会话/凭据服务，权限限制保留；没有提升权限或声称全局完全可见。所有可读自身进程及原VAE训练进程无安装包引用。两次Git原始字节预检也在删除前拒绝：先改为核直接原始输出，再为本次证据路径固定-text并实核Git原始字节，原始证明未重序列化冒充同SHA，全部失败记录保留。

2026-10-28T07:29:58.871419+08:00实际VAE56600尝试/56600Adam/0skip，原supervisor165374及worker165381的startticks一致，无exit、GPU锁仍被VAE持有。活动源码/spec/全部绑定哈希不变，旧57D仍T暂停。此次0新GPU/SDK/训练、0信号，未重启teacher/prefilter/旧协调器。继续自动VAE200k→实际导出与验收→LDM300k，不再询问开训。

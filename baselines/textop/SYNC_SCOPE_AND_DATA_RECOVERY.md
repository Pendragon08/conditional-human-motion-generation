## 2026-10-27追加：15000轮已真实核验并开始直接严格回放

15000轮完整状态与逐源统计已在服务器和D盘私有核验，公开推理30,208,218字节已上传Release textop-filter-recovery29d-20261027并核验digest。跟踪器停止、两次实际入口核验和全库直接回放代码/聚合结果本次增量同步；真实轨迹和逐源结果仍私有。历史“未来15k”说明已由这些实际收据覆盖。

# TextOp GitHub同步范围与数据恢复索引

本次按用户要求清点并增量同步TextOp复现代码、配置、报告和可公开聚合证据。
仓库为公开仓库，分支codex/textop-reproduction-20261024。
同步不改变当前训练，也不启动、停止或恢复任何实验。

同步前已有1844个版本控制文件，工作树合计约42.29MB：
scripts150份、provider_tools136份、configs58份、reports45份、
results1107份、upstream264份，以及环境文档和来源记录。
这些既有文件由Git增量机制复用，不重新上传整套内容。
新增84份历史本地辅助脚本共173603字节（只归档，不执行），
并补入服务器此前漏同步的一份2688字节碰撞审计源码。
服务器198份脚本/配置全部已覆盖：195份字节相同，3份可通过CRLF转LF精确恢复。
活动代码没有变更。逐文件摘要见results/github_sync_inventory_20261027。

## 已经在GitHub的资产

已通过GitHub API核对已有Release的资产尺寸和摘要元数据，无重新下载或上传：
- textop-filter-policy29d-20261026：1000/5000/10000轮的纯推理导出，约90.62MB合计。
- textop-quick-tracking-video-20261026：SFU科研跟踪视频与致谢，约7.00MB。
- textop-mvae-initial-20261024：历史57D检查点及说明，约513.35MB，不是当前69D模型。
- inactive-engineai-environment-cache-20261025、inactive-dreamcontrol-overlay-20261026：已有软件恢复归档，未重传。

具体链接、资产名、大小、摘要见existing_release_assets.json。
已有SHA核验收据及恢复说明保留；本次仅查API元数据，不冒称重新下载全量逐字节复验。

## 数据和私有结果保存范围

依据此前确定的许可与隐私要求，下列内容不向公开仓库传输：
AMASS/SMPL/BABEL原始或重定向数组、逐条来源/文本索引、真实逐试验轨迹、
含优化器及逐动作采样状态的完整检查点。
这些保存在原服务器本人目录与D:/具身智能/TextOp复现_20261024的私有备份目录。
仓库保存构建/恢复代码、聚合数量、备份摘要和验收收据；这不等于原始数据已上传GitHub。

本次仅做文件元数据清点：D盘checkpoints共15文件/3088144491字节，
datasets共140文件/7096959038字节，diagnostics共891文件/266856995字节，
videos共244文件/28658838字节。分类可能包含公开文件的私有留存副本；
这些数字不是重新校验所有私有内容的结果，也不是一次新归档。

主要恢复证据入口：
- 预处理：PREPROCESSING_BACKUP_AUDIT.md、reports/preprocessing_complete_20261026.md与after_kit_v1验收。
- 教师完整状态：FORMAL_CHECKPOINT_BACKUP.md及10k/恢复训练相应收据；完整状态私有、纯推理权重Release。
- 快速VAE/LDM：QUICK_VAE_ENVIRONMENT.md、QUICK_LDM_ENVIRONMENT.md及完整保存/私有备份收据。
- 已暂停全源VAE8100：GENERATOR_PAUSE_MARKER_ENVIRONMENT.md与results/generator_pause_20261027/private_backup_verified.json。
- 未来15k：FILTER_RECOVERY_CHECKPOINT_ENVIRONMENT.md及新run绑定工具；尚不存在时不能计为已同步。

## VPN与增量传输

本次不重传数据集、视频、模型或数GB软件归档。只上传缺失源码、最新聚合状态和小型清单。
提交前统计新Git对象包大小；该值是内容传输估计，不是VPN账单，
TLS/协议开销、代理路由和重试可能带来额外流量。最终实核commit和包大小记录于同步收据。

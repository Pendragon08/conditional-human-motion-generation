# AMASS 下载与人体模型安装进度

更新时间：2026-10-21T20:14:02.927471+08:00

本批15个动作子集：本地已核验 12/15，服务器已核验 10/15。另有此前已完成的 SSM。

动作压缩包总量为57.071 GB；SMPL-X模型压缩包为0.411 GB。TotalCapture依用户要求跳过。

后台下载进程：运行中（PID 51144）。

SMPL-X模型：服务器文件已核验；独立CPU模型前向：通过。

| 子集 | D盘压缩包 | 服务器文件 | 下载状态 |
|---|---|---|---|
| Transitions | 已核验 | 已核验 | SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED |
| ACCAD | 已核验 | 已核验 | SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED |
| BMLhandball | 已核验 | 已核验 | SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED |
| BMLmovi | 已核验 | 未完成 | SERVER_STORAGE_INSUFFICIENT |
| BMLrub | 未完成（已接收3202.7 MB） | 未完成 | DOWNLOADING |
| CMU | 未完成（已接收5582.6 MB） | 未完成 | DOWNLOADING |
| DFaust | 已核验 | 已核验 | SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED |
| EKUT | 已核验 | 已核验 | SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED |
| EyesJapanDataset | 已核验 | 未完成 | SERVER_STORAGE_INSUFFICIENT |
| HDM05 | 已核验 | 已核验 | SERVER_VERIFIED_AFTER_CLIENT_ERROR |
| HumanEva | 已核验 | 已核验 | SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED |
| KIT | 未完成（已接收151.0 MB） | 未完成 | DOWNLOADING |
| MoSh | 已核验 | 已核验 | SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED |
| PosePrior | 已核验 | 已核验 | SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED |
| SFU | 已核验 | 已核验 | SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED |

原始包位于本文件夹的 `archives/`。模型已解压到 `body_models_smplx_lockedhead_20230207/`。

服务器接收下一包时保留5 GiB余量。后续删除必须先完成GitHub备份核验；AMASS和SMPL-X原始文件保留。此前清理的服务器压缩副本在D盘有完整校验备份。

表中“已核验”指已完成归档CRC、哈希与服务器解压校验；服务器旧压缩副本可能已清理，完整包保留在D盘。这不代表所有动作已通过数值检查或完成重定向。

完整AMASS未下载齐；本轮G1配对动作、262D序列均为0，生成器训练未开始。SMPL-X独立检查只确认模型加载与前向计算。

AMASS及SMPL-X原始数据和模型权重不上传GitHub。仓库保存获取代码、清单、报告及验证证据。

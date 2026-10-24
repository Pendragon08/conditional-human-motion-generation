# AMASS 下载与人体模型安装进度

更新时间：2026-10-21T19:43:07.199061+08:00

本批15个动作子集：本地已核验 11/15，服务器已核验 9/15。另有此前已完成的 SSM。

动作压缩包总量为57.071 GB；SMPL-X模型压缩包为0.411 GB。TotalCapture依用户要求跳过。

后台下载进程：已结束；请检查各项结果（PID 64876）。

SMPL-X模型：服务器文件已核验；独立CPU模型前向：通过。

| 子集 | D盘压缩包 | 服务器文件 | 下载状态 |
|---|---|---|---|
| Transitions | 已核验 | 已核验 | SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED |
| ACCAD | 已核验 | 已核验 | SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED |
| BMLhandball | 已核验 | 已核验 | SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED |
| BMLmovi | 已核验 | 未完成 | SERVER_STORAGE_INSUFFICIENT |
| BMLrub | 未完成 | 未完成 | NEEDS_ATTENTION |
| CMU | 未完成 | 未完成 | NEEDS_ATTENTION |
| DFaust | 已核验 | 已核验 | SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED |
| EKUT | 已核验 | 已核验 | SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED |
| EyesJapanDataset | 未完成 | 未完成 | NEEDS_ATTENTION |
| HDM05 | 已核验 | 未完成 | SERVER_STORAGE_INSUFFICIENT |
| HumanEva | 已核验 | 已核验 | SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED |
| KIT | 未完成 | 未完成 | NEEDS_ATTENTION |
| MoSh | 已核验 | 已核验 | SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED |
| PosePrior | 已核验 | 已核验 | SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED |
| SFU | 已核验 | 已核验 | SERVER_ARCHIVE_AND_EXTRACTED_FILES_VERIFIED |

原始包位于本文件夹的 `archives/`。模型已解压到 `body_models_smplx_lockedhead_20230207/`。

服务器仅在可同时容纳压缩包、解压文件并保留5 GiB余量时接收下一包；余量不足会保留本地完整包，不删除现有数据。

表中“已核验”指归档CRC、哈希与服务器解压文件记录，不代表所有动作已通过数值检查或完成重定向。

完整AMASS未下载齐；本轮G1配对动作、262D序列均为0，生成器训练未开始。SMPL-X独立检查只确认模型加载与前向计算。

AMASS及SMPL-X原始数据和模型权重不上传GitHub。仓库保存获取代码、清单、报告及验证证据。

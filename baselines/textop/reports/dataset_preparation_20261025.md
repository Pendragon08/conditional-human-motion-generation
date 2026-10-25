# TextOp 数据准备核查与重建记录

更新时间：2026-10-25 16:41:10 北京时间。

**论文 Step1 尚未完整复现；生成器参数更新已暂停，保留训练现场与检查点，先完成数据准备。**

已按用户2026-10-25的新指令记录后续顺序：数据验收 → VAE → LDM → 历史2帧生成未来8帧的自回归验证。见[Step2训练安排](step2_motion_generator_20261025.md)。[69D核心数值验证已通过](paper69_runtime_validation_20261025.md)，完整数据加载/归一化集成及物理筛选仍未完成；原57D训练现场继续暂停。

| 用户所列环节 | 已实际完成 | 尚未完成/差异 |
| --- | --- | --- |
| AMASS、BABEL、自建数据 | 全量读取公开RobotMDAR配对包8,291条；原始stageii索引初次精确匹配500条；修复文件命名后又确认143条已有输入；另从本地完整压缩包按索引提取缺失输入。BABEL官方原始包已独立下载并安装到服务器；8,291条运动对应50,004条分段标注的ID、来源、文本、类别与时间均核对一致 | 首轮未匹配7,791条并不都需要重下；最新覆盖见下方记录。作者私有数据不可用；标签一致尚不等于全部动作已完成物理筛选 |
| GMR人体→机器人 | 实际执行SMPL-X前向与GMR29关节IK；首批500条及补充143条均已处理完成；新增1653条已完成1653条；CMU接续处理901条 | 旧公开包是23DoF/57D；不能补6个零就声称恢复论文29DoF/69D |
| 过滤、对齐、训练集构建 | 全量结构与划分核查、时间标注修整视图、29DoF接触/镜像/69D编码已运行 | 论文要求的真实策略筛选还未执行，全部新输出physical_filter_passed=false、训练就绪=false |

## 论文与固定公开版本的差异

[论文附录A及表VI](https://arxiv.org/pdf/2602.07439#page=13)给出29DoF、69D、30→50Hz、接触高度0.2m；固定公开脚本裁为23DoF/57D、接触高度0.08m。论文筛选需训练使用特权信息且无域随机化的跟踪策略4万步，先移除高度终止项并剔除失败率大于0.05的动作，再恢复标准终止条件、剔除任何失败动作。原来的“已下载”不能代表这些步骤已被我们重新执行。

作者私有数据和部分最新实现未公开。本次不编造替代数据，也不把小批重建或文件校验当成全规模物理筛选。过滤滑动平均的精确窗口和严格评估重复次数未明确公开，配置中留空，后续采用的选择必须单独标注。

## 公开数据全量核查

- train6,213、val2,078；原始包合计50,004条分段文本标注。所有运动数组有限、50Hz、23关节，文本嵌入缺失数0。
- 训练/验证的源路径、BABEL ID、完整运动指纹交叉数均为0；训练集中存在2个重复运动指纹，未静默删除。
- 7,635条声明长度与实际数组长度不同；官方加载器本就按motion_len修正，这不等同于7,635条运动损坏。
- 4,988条标注末端超过对应运动可用时段；另有5条无效时间标注和1条无运动重叠。已生成独立时间对齐视图，保留原值与排除理由，原PKL未修改。
- 实际时长为105,442.66秒；发布包比README多6条序列，已如实记录。

对齐视图结果：
```json
{
  "scope": "aligned metadata view of the released23DoF data; not the paper69D dataset",
  "source_arrays_modified": false,
  "source_pkl_modified": false,
  "physical_filter_reexecuted": false,
  "counts": {
    "retained_pairs": 49998,
    "intervals_clipped": 4987,
    "train_sequences": 6213,
    "invalid_interval_removed": 5,
    "no_overlap_removed": 1,
    "val_sequences": 2078
  }
}
```

## 29DoF 重建进度

当前成功 500/500 条、失败 0 条，并分别保存对应镜像；这些是未通过动态过滤的候选数据。

补充143条：成功143、失败0。新增1653条：成功1653、失败0。CMU另批成功901条、失败0条。不计重复的6条smoke验证，合计已完成3197条候选运动、失败0条；保存的镜像另计。新批次原始输入已逐文件校验，RAM仅作服务器工作副本。

CMU完整原始包已通过CRC/tar校验；其选定SMPL-X G版本覆盖1,343条所需索引中的1,330条，另13条来源在该包内不存在，已保留私有缺失清单，未计为已处理。1,330条相关输入直接流式传输到个人RAM工作目录，逐文件SHA256核验后，等待前一批完成且持久归档成功，再串行执行同一GMR代码；不打断当前任务。当前CMU调度状态：
```json
{
  "status": "RUNNING_GMR_CPU_ONLY",
  "time_beijing": "2026-10-25T16:41:08.996901+08:00",
  "target": 1330,
  "requested_indexed_sources": 1343,
  "missing_from_selected_official_export": 13,
  "raw_bytes": 18804784599,
  "archive_sha256": "8a276bd0ed23fa72202f80f1dcf8fd9a336eb053ad86233cd47a9a2f51210270",
  "source_durable": "complete C-drive original archive; private original-BABEL-verified index on D",
  "ram_working_copy": true,
  "physical_filter_executed": false,
  "staged_files": 1300,
  "staged_bytes": 18347165831,
  "verification": {
    "files_sha256_verified": 1330,
    "raw_bytes": 18804784599
  },
  "launch": {
    "pid": 3483711,
    "command": [
      "/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python",
      "/data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/prepare_paper_data.py",
      "--spec",
      "/data/code/ouyanghao/oyh/textop-reproduction-20261024/configs/data-prep-env-spec.json",
      "--pairs",
      "/dev/shm/ouyanghao_textop_20261025/cmu1330_v1/pairs.private.json",
      "--output",
      "/dev/shm/ouyanghao_textop_20261025/cmu1330_v1/prepared1330_v1"
    ],
    "cpu_only": true,
    "source_sha256": "90db2afdb9baf3521356c7995c95d2cd48bdc56836ffc4539073c4c8e4b819dc",
    "target": 1330,
    "time_utc": "2026-10-25T07:07:44Z"
  },
  "progress": {
    "spec_sha256": "6e7f21352c2839d62498a068103cfedcd3949f5416b1ca075477f6bdbef94591",
    "versions": {
      "numpy": "1.26.4",
      "torch": "2.5.1+cu121",
      "scipy": "1.15.3",
      "mujoco": "3.3.7",
      "mink": "0.0.13",
      "smplx": "0.1.28",
      "qpsolvers": "4.8.1",
      "daqp": "0.7.2",
      "rich": "13.9.4",
      "tqdm": "4.70.0"
    },
    "target_dof": 29,
    "feature_dim": 69,
    "output_fps": 50,
    "completed_sequences": 901,
    "mirrored_sequences": 901,
    "failed_sequences": 0,
    "total_requested": 1330,
    "physical_filter_passed": false,
    "ready_for_generator_training": false,
    "status": "RUNNING",
    "cpu_only": true,
    "implementation": "own paper-oriented preparation, GMR IK reused; not authors private final pipeline",
    "processed": 901,
    "updated_utc": "2026-10-25T08:41:02Z"
  }
}
```

另外143条已有ACCAD原始动作因空格/下划线导出命名差异，首轮未匹配；现通过明确的字面规范化确认，无模糊匹配、无歧义键。当前批次完成后使用同一验证过的CPU环境串行处理，已完成 143/143 条。输入可用性不等于完成重定向或通过物理筛选。

```json
{
  "sequences": 500,
  "frames50": 429871,
  "paired_annotations_including_saved_mirrors": 9396,
  "rejected_annotations": 122,
  "codec_max_error_m_or_rad": 3.981501353056416e-07,
  "joint_skeleton_mirror_max_error_m": 9.99999971451615e-06,
  "splits": {
    "train": 373,
    "val": 127
  }
}
```

处理链：已授权SMPL-X原始动作 → 保留16个形状参数的人体前向 → 复用GMR核心IK → 保留全部29关节 → 按实际秒轴重采样50Hz → 按论文阈值重算足接触 → 对齐BABEL派生标注 → 骨架和方向词镜像 → 69D编码/逆变换核查。

BMLrub与KIT的后续队列已安排：分别等待完整原始包CRC/SHA校验、前一批GMR与持久备份成功，再逐一精确核对索引、串行重定向和归档。缺失或有歧义的来源保留清单，不填造动作、不启动重叠GMR。当前两项成功条数分别为0、0，下载/排队本身不计入已处理数据。

最新独立核对：两包均已完成CRC/tar/SHA验证；BMLrub2,139/2,139、KIT2,513/2,513条索引精确唯一匹配，零缺失、零歧义。所有可用公开配对源为8,278/8,291条，差额仅为CMU13条；与论文12,296条公共跟踪运动及403条私有运动的范围差异单列，不冒充完整作者语料。

SMPL-X与GMR部分复用已有授权资产/代码；时间轴、接触、镜像、69D编解码与调度为本次自实现，明确区别于官方旧处理脚本。GMR源码镜像没有可用Git提交，使用源文件和配置SHA256固定。CPU环境依赖未修改，已有独立按文档执行的数值验证。

首轮镜像检查错误地包含固定IMU安装框架，其真实横向偏置会产生4.48mm镜像残差；已改为核查动作表示中的根与驱动关节骨架，未提高阈值掩盖误差。首轮失败记录保留。SMPL-X批量前向的表情批维适配错误也已修复并保留日志。

## 原始动作覆盖与下一步

初次严格路径匹配未定位的原始动作按数据子集分布（历史核查结果，后续命名修复和本地提取见下）：
```json
{
  "BMLrub": 2139,
  "ACCAD": 143,
  "CMU": 1343,
  "EyesJapanDataset": 542,
  "KIT": 2513,
  "BMLmovi": 1111
}
```

现有服务器原始输入的命名修复记录：
```json
{
  "release_sequences": 8291,
  "previous_exact_raw_matches": 500,
  "additional_existing_raw_matches": 143,
  "matched_by_group": {
    "ACCAD": 147,
    "HDM05": 165,
    "EKUT": 220,
    "DFaust": 79,
    "SFU": 32
  },
  "still_unmatched_by_group": {
    "BMLrub": 2139,
    "CMU": 1343,
    "EyesJapanDataset": 542,
    "KIT": 2513,
    "BMLmovi": 1111
  },
  "ambiguous_normalized_keys": 0,
  "newly_retargeted_sequences": 0,
  "physical_filter_executed": false,
  "note": "Additional input availability is not completed retargeting or dynamic filtering"
}
```

已核实D盘原始压缩包可复用。按文本索引从BMLmovi、EyesJapan完整包提取到C盘独立私有目录，目标1,653条；不重复下载、不修改压缩包、保留6GiB系统盘余量。完整压缩包先与此前保存的SHA256比对，逐文件再保存SHA256；不能将未提取或尚未转移到服务器的动作计为就绪数据。
```json
{
  "requested": 1653,
  "completed": 1653,
  "bytes": 18369027723,
  "groups": {
    "BMLmovi": {
      "archive_bytes": 5295135910,
      "archive_sha256": "b8c5fca189bdb52768cdef8b93a3892a9e0788100b4553aa1ec63eac1ddf66c0",
      "matched": 1111,
      "bytes": 6223276696
    },
    "EyesJapanDataset": {
      "archive_bytes": 7628321177,
      "archive_sha256": "85c83ca46fad11c71e4749042d3e4d0cabf21238966c3c41924016d9c53f25cf",
      "matched": 542,
      "bytes": 12145751027
    }
  },
  "status": "LOCAL_EXTRACTION_COMPLETE",
  "new_downloads": 0,
  "physical_filter_executed": false,
  "updated_utc": "2026-10-24T18:41:17Z",
  "unmatched_requested": 0
}
```

新提取的1,653条输入后续串行处理队列（同一已验证CPU环境，不启动生成器或RL）：
```json
{
  "status": "PREPROCESSING_BACKED_UP_FILTER_PENDING",
  "target": 1653,
  "ram_working_copy": true,
  "source_durable": "D complete original archives and verified private recovery kit; RAM is working copy",
  "physical_filter_executed": false,
  "time_utc": "2026-10-25T07:07:13.469000+00:00",
  "preflight": {
    "mem": {
      "MemTotal": 269956653056,
      "MemAvailable": 247712681984
    },
    "raw_bytes": 18369027723,
    "ram_root": "/dev/shm/ouyanghao_textop_20261025/local1653_v1"
  },
  "launch": {
    "pid": 3319567,
    "command": [
      "/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python",
      "/data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/prepare_paper_data.py",
      "--spec",
      "/data/code/ouyanghao/oyh/textop-reproduction-20261024/configs/data-prep-env-spec.json",
      "--pairs",
      "/dev/shm/ouyanghao_textop_20261025/local1653_v1/pairs.private.json",
      "--output",
      "/dev/shm/ouyanghao_textop_20261025/local1653_v1/prepared1653_v1"
    ],
    "verified_raw": 1653,
    "source_sha256": "90db2afdb9baf3521356c7995c95d2cd48bdc56836ffc4539073c4c8e4b819dc",
    "time_utc": "2026-10-25T00:45:18Z",
    "durability": "raw originals in localC and archivesD; RAM is working copy; outputs must be backed up to localD"
  },
  "progress": {
    "spec_sha256": "6e7f21352c2839d62498a068103cfedcd3949f5416b1ca075477f6bdbef94591",
    "versions": {
      "numpy": "1.26.4",
      "torch": "2.5.1+cu121",
      "scipy": "1.15.3",
      "mujoco": "3.3.7",
      "mink": "0.0.13",
      "smplx": "0.1.28",
      "qpsolvers": "4.8.1",
      "daqp": "0.7.2",
      "rich": "13.9.4",
      "tqdm": "4.70.0"
    },
    "target_dof": 29,
    "feature_dim": 69,
    "output_fps": 50,
    "completed_sequences": 1653,
    "mirrored_sequences": 1653,
    "failed_sequences": 0,
    "total_requested": 1653,
    "physical_filter_passed": false,
    "ready_for_generator_training": false,
    "status": "PREPROCESSING_FINISHED_FILTER_PENDING",
    "cpu_only": true,
    "implementation": "own paper-oriented preparation, GMR IK reused; not authors private final pipeline",
    "processed": 1653,
    "updated_utc": "2026-10-25T07:04:36Z",
    "splits": {
      "train": 1236,
      "val": 417
    },
    "paired_segments": 24158
  },
  "observer_recovered": true,
  "prior_observer_error_saved_in": "local1653_queue_error_before_resume.json",
  "archive_sha256": "ed60f1458ce1325fff85cbea07d651063f265423dda776ef349a37191f862d7a",
  "archive_bytes": 1229573239,
  "per_motion_backup_sha256_verified": 1653
}
```

队列仅在本地完整提取且前面500+143条批次成功结束后运行；先逐文件校验，再检查服务器可用内存及共享内存容量，使用个人权限目录中的RAM工作副本进行GMR处理，最终编码结果归档到D盘并校验。D盘完整许可压缩包和私有索引是持久输入；C盘重复解压缓存已经过原始包哈希、服务器全量副本及实际恢复验证后清理，恢复工具和证据先备份到GitHub。RAM工作副本不作为唯一数据副本。原始包、处理后数据、检查点与实验记录均保留。

本轮用户已回复AMASS登录完成，并明确同意BABEL独立许可及下载。CMU独立下载会话已完成官网文件主机认证；本次采样状态为LOCAL_ORIGINAL_ARCHIVE_VERIFIED。BABEL官方原始包已独立下载并安装到服务器；8,291条运动对应50,004条分段标注的ID、来源、文本、类别与时间均核对一致。原始标注位于D盘个人datasets/BABEL及服务器data_preparation_20261025/babel_v1_0目录，已核对SHA256；不公开发布原始标注。详见[BABEL独立核查](babel_original_alignment_20261025.md)。

根据三个下载器当前校验凭据，BMLrub、CMU、KIT合计尚需续传约0.00GiB；完整校验通过的包无需重复下载。当前C盘余6.15GiB、D盘余8.81GiB。继续为处理结果和检查点保留余量；服务器容量另行核查。现有GMR队列继续运行，TotalCapture仍跳过。

重复缓存清理记录（全部原始数据包、已处理数据、检查点、唯一实验记录均保留）：
```json
{
  "time_beijing": "2026-10-25T10:46:06.646061+08:00",
  "status": "REDUNDANT_EXTRACTION_CACHE_EVICTED",
  "files_unlinked": 1653,
  "recoverable_logical_bytes": 18369027723,
  "previous_cache_physical_bytes": 13103345664,
  "C_free_before": 18490896384,
  "C_free_after": 31591403520,
  "recovery_code_and_evidence_backed_up_github_commit": "a730c72ed63c5362964042e17eb1ddb739ce123c",
  "original_dataset_archives_deleted": 0,
  "prepared_dataset_files_deleted": 0,
  "checkpoints_deleted": 0,
  "experiment_records_deleted": 0,
  "private_indexes_retained": true,
  "server_working_copy_retained": true,
  "restore_command": "restore_indexed_cache.py --index <retained ready.private.json> --archive-dir <retained original archives> --output <private cache> --limit <optional count>",
  "restore_helper_sha256": "1c6f6d3d6b6233e2b4bc88337115d9d488ef829e7d219791a32ba181d6203183"
}
```

授权和空间记录：
```json
{
  "recorded_beijing": "2026-10-25T14:24:14.452056+08:00",
  "amass": {
    "user_reported_login_complete": true,
    "observed_main_download_table_accessible": true,
    "file_host_session_verified_this_turn": true,
    "existing_approved_subsets_remain_authorized": true,
    "totalcapture_skipped": true
  },
  "babel": {
    "user_explicit_license_authorization": true,
    "license_url": "https://babel.is.tue.mpg.de/license.html",
    "scope": "non-commercial research; personal reproduction storage; no redistribution",
    "observed_download_redirect": null,
    "download_completed": true,
    "status": "OFFICIAL_BABEL_V1_0_DOWNLOADED_ZIP_CRC_AND_JSON_VERIFIED",
    "user_reported_login_complete": true,
    "observed_download_page": "https://babel.is.tue.mpg.de/download.php",
    "archive_sha256": "863d79b78f2d0225df74140b94cd61e06e04217726ae4ba1d2f50bad0d902428",
    "server_installed": true,
    "original_alignment_checked": true
  },
  "storage": {
    "C_free_bytes": 7180431360,
    "D_free_bytes": 10692722688,
    "remaining_archives": [
      {
        "subset": "BMLrub",
        "expected_archive_bytes_from_prior_official_manifest": 12919496284,
        "current_partial_bytes": 12919496284,
        "remaining_bytes_estimate": 0
      },
      {
        "subset": "CMU",
        "expected_archive_bytes_from_prior_official_manifest": 11572134418,
        "current_partial_bytes": 11572134418,
        "remaining_bytes_estimate": 0
      },
      {
        "subset": "KIT",
        "expected_archive_bytes_from_prior_official_manifest": 13182414052,
        "current_partial_bytes": 13182414052,
        "remaining_bytes_estimate": 0
      }
    ],
    "total_additional_archive_bytes_estimate": 0,
    "status": "ASSETS_RESUMING_WITH_6GIB_PER_VOLUME_RESERVE",
    "requested_free_capacity_gb": 50,
    "download_states": {
      "BMLrub": "LOCAL_ORIGINAL_ARCHIVE_VERIFIED",
      "CMU": "LOCAL_ORIGINAL_ARCHIVE_VERIFIED",
      "KIT": "LOCAL_ORIGINAL_ARCHIVE_VERIFIED"
    }
  },
  "credentials_recorded": false,
  "new_raw_assets_published": false
}
```

数据盘约剩4.60GiB、系统盘约剩6.23GiB。继续完成新增1653条并核验持久备份，保留失败理由；补齐原始输入后，准备未过滤的跟踪数据并运行论文两轮真实评估。只有通过动态筛选的动作才能进入后续论文69D生成器训练。旧57D检查点不作为69D检查点继续使用。

代码与聚合证据同步GitHub；包含原始动作、人体模型、逐条文本的资产遵循许可保存在个人目录和本地，GitHub不公开原始/重定向数据文件。新输出均明确标为待物理筛选。PhyGile、ReactiveBFM继续暂停，未改动其数据或实验记录。

服务器数据目录：/data/code/ouyanghao/oyh/textop-reproduction-20261024/data_preparation_20261025。

训练现场：State:	T (stopped)。

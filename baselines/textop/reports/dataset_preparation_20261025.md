# TextOp 数据准备核查与重建记录

更新时间：2026-10-25 02:15:51 北京时间。

**论文 Step1 尚未完整复现；生成器参数更新已暂停，保留训练现场与检查点，先完成数据准备。**

| 用户所列环节 | 已实际完成 | 尚未完成/差异 |
| --- | --- | --- |
| AMASS、BABEL、自建数据 | 全量读取公开RobotMDAR配对包8,291条；原始stageii索引初次精确匹配500条；修复文件命名后又确认143条已有输入；另从本地完整压缩包按索引提取缺失输入 | 首轮未匹配7,791条并不都需要重下；最新覆盖见下方记录。BABEL标注本轮复用官方包内字段，未独立重下原始BABEL JSON；作者私有数据不可用 |
| GMR人体→机器人 | 实际执行SMPL-X前向与GMR29关节IK；6条完整动作通过处理验证；500条批处理及后续143条串行队列已启动 | 旧公开包是23DoF/57D；不能补6个零就声称恢复论文29DoF/69D |
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

当前成功 422/500 条、失败 0 条，并分别保存对应镜像；这些是未通过动态过滤的候选数据。

另外143条已有ACCAD原始动作因空格/下划线导出命名差异，首轮未匹配；现通过明确的字面规范化确认，无模糊匹配、无歧义键。当前批次完成后使用同一验证过的CPU环境串行处理，已完成 0/143 条。输入可用性不等于完成重定向或通过物理筛选。

```json
{
  "sequences": 422,
  "frames50": 194900,
  "paired_annotations_including_saved_mirrors": 4884,
  "rejected_annotations": 121,
  "codec_max_error_m_or_rad": 3.981501353056416e-07,
  "joint_skeleton_mirror_max_error_m": 9.999999626392198e-06,
  "splits": {
    "train": 315,
    "val": 107
  }
}
```

处理链：已授权SMPL-X原始动作 → 保留16个形状参数的人体前向 → 复用GMR核心IK → 保留全部29关节 → 按实际秒轴重采样50Hz → 按论文阈值重算足接触 → 对齐BABEL派生标注 → 骨架和方向词镜像 → 69D编码/逆变换核查。

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
  "completed": 1180,
  "bytes": 7892851751,
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
      "matched": 69,
      "bytes": 1669575055
    }
  },
  "status": "EXTRACTING_INDEXED_MOTIONS",
  "new_downloads": 0,
  "physical_filter_executed": false,
  "updated_utc": "2026-10-24T18:15:45Z"
}
```

新提取的1,653条输入后续串行处理队列（同一已验证CPU环境，不启动生成器或RL）：
```json
{
  "status": "WAITING_FOR_LOCAL_EXTRACTION_AND_PREVIOUS_RETARGETING",
  "target": 1653,
  "ram_working_copy": true,
  "source_durable": "C private assets and D licensed original archives",
  "physical_filter_executed": false,
  "time_utc": "2026-10-24T18:03:18.584551+00:00"
}
```

队列仅在本地完整提取且前面500+143条批次成功结束后运行；先逐文件校验，再检查服务器可用内存及共享内存容量，使用个人权限目录中的RAM工作副本进行GMR处理，最终编码结果归档到D盘并校验。C盘原始选中动作和D盘许可压缩包是持久输入，RAM工作副本不作为唯一数据副本。若资源条件不满足，保留记录并停止；不会删除数据或重启旧训练。

AMASS官网下载当前显示Sign In，已请求用户重新登录以续传BMLrub、CMU、KIT等缺少的包。BABEL原始标注另需独立登录和许可确认；已有AMASS/SMPL-X授权不自动覆盖BABEL。本次尚未接受BABEL独立许可，也未发送作者邮件。TotalCapture按此前要求跳过。

数据盘约剩1.55GiB、系统盘约剩1.03GiB。先完成已有500条的处理并保留失败理由；补齐原始输入后，准备未过滤的跟踪数据并运行论文两轮真实评估。只有通过动态筛选的动作才能进入后续论文69D生成器训练。旧57D检查点不作为69D检查点继续使用。

代码与聚合证据同步GitHub；包含原始动作、人体模型、逐条文本的资产遵循许可保存在个人目录和本地，GitHub不公开原始/重定向数据文件。新输出均明确标为待物理筛选。PhyGile、ReactiveBFM继续暂停，未改动其数据或实验记录。

服务器数据目录：/data/code/ouyanghao/oyh/textop-reproduction-20261024/data_preparation_20261025。

训练现场：State:	T (stopped)。

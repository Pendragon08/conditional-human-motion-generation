# PhyGile 配对数据准备状态（2026-10-18）

**已取得官方公开文本、划分和来源索引；尚未取得对应 G1 动作训练集，配对机器人序列数为 0，未开始生成器训练。** 此状态是上轮“缺配对数据”的细化，不表示 HumanML3D 没有文本标注。

## 本次实际取得

固定官方 HumanML3D 提交 `9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23`，下载 README、LICENSE、index.csv、texts.zip 与五份划分文件，共 9 个文件，逐文件 SHA-256 见 `results/paired_data_20261018/public_metadata_manifest.json`。

- texts.zip：29,232 个文本文件，87,384 条非空标注。
- 来源索引：14,616 行，字段为 source_path、start_frame、end_frame、new_name。
- 实际官方划分：train 23,384、val 1,460、test 4,384，合计 29,228 个 ID。
- 文本包比 all.txt 多 4 个 ID，按官方划分处理，不把额外条目静默并入训练。
- ZIP CRC、划分并集、文本覆盖、原始索引覆盖及去镜像后的动作 ID 隔离均核验通过。时间标签异常条数 200，逐行位置保留在 `alignment_inventory.json`，需要在后续配对时显式处理，尚未据此生成任何动作。
- 按 index.csv 的 source_path 核对，原始来源文件跨划分重叠数：{'train_val': 224, 'train_test': 639, 'val_test': 49}。动作 ID 隔离不等同于受试者隔离；不能据此声称未见受试者泛化。

## 论文所需的配对关系

PhyGile §III-B 使用经过筛选、重定向到机器人形态的 HumanML3D 动作与原文本配对。文本无须全部重新标注。需要按同一 ID、原始录制、时间范围和镜像关系处理：

`HumanML3D 文本及索引 + 对应 AMASS 动作 → GMR 重定向到 G1 → 时间/语义与运动学核对 → 本实现的 262D 编码 → 训练 manifest`

保留官方 train/val/test 与镜像对应关系，并核对原始录制来源；子片段标注必须按时间对齐，不能把整句标签赋给任意裁剪片段。HumanML3D 的人体特征不能通过删改一列直接作为 G1 262D 数据。

官方动作 ID 划分存在上述原始来源文件重叠，不能同时宣称“原样官方划分”和“原始录制文件完全隔离”。现有训练入口会拒绝 source_group 跨划分重叠，后续必须明确分组与实验协议；若改变官方划分，应记录与原论文的差异。本次仅准备元数据，没有据此创建或启动研究训练。

## 当前具体缺什么

1. 与索引对应的原始 AMASS 动作文件。AMASS 官网下载页需要账号登录。
2. 所选重定向路径所需的人体模型。当前官方 GMR 的 AMASS 路线使用 SMPL-X 动作和 SMPL-X 模型文件，并明确提示不要直接使用 SMPL+H 输入。SMPL-X 官网要求注册并同意许可。
3. 实际机器人动作生成、时间对齐、过滤与 262D 提取；这些步骤尚未在本次公开文本上执行，不能把文本下载当作完整配对数据已就绪。

HumanML3D 官方原始重建脚本使用 SMPL+H/DMPL；它与 GMR 的 SMPL-X 入口不是同一格式，需选择并核对一致的处理链。没有把第三方镜像冒充作者发布的 G1 数据包。本次检查 PhyGile 项目及跟踪仓库，未找到作者筛选后的 G1 动作—文本配对包下载入口。

当前 20/4 个武术参考没有对应的 HumanML3D 原始文本，仍用于跟踪验证，不当作该生成器的完整配对训练集。

## 来源

- [PhyGile 原文 §III-B](https://arxiv.org/html/2603.19305v3)
- [PhyGile 官方项目](https://baojch.github.io/phygile-page/)
- [HumanML3D 官方获取说明](https://github.com/EricGuo5513/HumanML3D#how-to-obtain-the-data)
- [HumanML3D 官方文本及划分目录](https://github.com/EricGuo5513/HumanML3D/tree/9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23/HumanML3D)
- [AMASS 下载入口](https://amass.is.tue.mpg.de/download.php)
- [SMPL-X 官网](https://smpl-x.is.tue.mpg.de/)
- [GMR Data Preparation](https://github.com/YanjieZe/GMR#data-preparation)

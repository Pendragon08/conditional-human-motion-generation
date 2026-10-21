# HumanML3D 索引动作到 G1 配对特征（2026-10-21）

全量 AMASS 下载已依用户要求停止，所有完整包及断点保留。后续处理依据 HumanML3D 来源索引，不按官网全部数据规模扩张。

## 来源和时间对应

固定 HumanML3D 版本 `9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23` 的文本、index.csv 及 train/val/test。索引要求11,713个独立录制；其中 SSM 有30个录制、60个动作ID（含镜像）。其他来源的覆盖和缺口见 `results/humanml_g1_pipeline_20261021/indexed_inventory/`，匹配文件名不代表时间对应已验证。

为验证 SSM，另外仅取得其 SMPL+H 参考包（13,532,334字节，SHA256 `f3746e0d6f44354ae08d075208d47f5b5564934667429d56c6f2d21fcd376929`），沿用用户已接受的 SSM 许可。SMPL+H仅用于原始索引及测量标记点核验；GMR加载的是SMPL-X动作和兼容SMPL-X模型。

SSM两个导出的同名公共测量标记点逐帧相同，其中13个录制的采样率标头不同：SMPL+H约59.9999886 Hz，SMPL-X为120 Hz；另17个录制的参考标头约120.000046 Hz。处理程序用测量标记点验证逐帧身份，并遵循官方HumanML3D的 `int(reference_fps/20)` 抽帧、数据集前缀裁剪、index.csv裁剪、文本20Hz子片段裁剪的顺序。保留原始帧号和名义20Hz约定，不使用GMR的linspace重采样来替换文本时间轴。

标记点缺失在两种导出中分别可能表示为全零XYZ或NaN；仅归一这两种缺失表示，其余有效测量值须一致。至少30个同名标记点，每帧至少10个有效同名点，最大坐标差不得超过1e-7米。不能通过这些检查的来源不产生配对数据。

## 动作处理和边界

复用服务器现有 `gmr_amass_20261021@141547bf` CPU环境，未修改任何共享包。调用现有GMR求解器，显式按关节名将29个G1关节映射到已有骨架。SMPL-X显式使用16个beta及原始手、眼、颌姿态；原始文件未提供表情系数，显式使用零表情。

镜像ID使用G1根位姿几何反射、左右关节交换及URDF轴方向符号；使用原本的M前缀文本，不重复交换文字。随机16姿态核验显示左右URDF的小数偏差导致最大约0.01毫米位置不对称，容差显式为0.02毫米，不声称机器人几何完全对称。

筛选阈值在 `configs/humanml_pairing.json` 中提前固定，是独立复现选择，不是作者发布的阈值。只覆盖有限性、最小帧数、关节限位、跳变、根部速度及身体坐标原点穿地。**不等同于网格碰撞、力矩约束、动力学可执行性或GMT跟踪成功验证。** 262D采用现有版本化独立codec；不是已确认与作者逐通道完全一致的表示。

同一动作的整段文本及每个有效子片段分别保存。保留官方ID划分，并另外生成排除跨划分原始录制的manifest。归一化仅拟合后者的训练集。所有真实逐帧数据保存在用户服务器和D盘，不公开上传。

## 独立按文档核验命令

以下命令在原服务器执行。它只读取已有数据并创建一份新的核验回执，不训练、不删除、不修改环境。输出文件必须不存在：

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/scripts/audit_humanml_g1_dataset.py --dataset /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/ssm_batch_v1 --public /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public --skeleton /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/configs/g1_codec_skeleton.json --gmr-root /data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/GMR --output /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/ssm_dataset_audit_document_r2.json
```

成功须退出0且状态为 `DATASET_CONSISTENCY_CHECKS_PASSED`。预期核对66条序列、5,904帧、182条文本和30个原始录制；重算文本帧范围、原始文件与特征哈希、codec往返、MuJoCo/FK、录制划分隔离和训练集归一化，并实际调用生成器的数据加载器。作者运行另存 `ssm_dataset_audit_author.json`，不能替代独立文档回执。

核验人另保存执行记录，绑定本文档执行前后SHA256、原样命令、UTC起止时间、退出码、标准输出/错误文件及回执的SHA256。执行期间文档哈希必须不变。先前 `ssm_dataset_audit_document.json` 及原始记录保留；其执行后文档发生过说明性修改，只作为历史核验，不替代本版本的文档核验。

这批数据只有SSM，且没有验证集样本；不代表完整HumanML3D、完整PhyGile训练集或论文性能。完整生成器训练尚未开始。

## 文件保留要求

用户要求所有数据、代码和实验结果在GitHub备份核验前不删除。当前没有删除任务。对许可禁止公开分发的AMASS、人体模型和逐帧衍生数据，保留在授权本地/服务器，GitHub只存代码、清单、哈希、统计和报告。

# 修正版 29DoF 条件扩散入口

此目录独立于 2026-11-03 已归档模型、四类参考和 PPO 作业。训练从原 MDM 初始化开始，使用同一 2,091 条训练、484 条验证及 586 条测试的归档数据和训练统计，首轮固定 10,000 次优化器更新。输出使用新阶段目录、run 名和新检查点，旧源文件继续由哈希绑定。

## 训练与采样

`train.py` 从既有训练入口单独加载一个模块实例，并仅在该实例上接入新 batch、loss、guide 和见证回调。数据审计、模型构建、CLIP 冻结、optimizer、累计梯度、磁盘门槛及检查点保存沿用既有实现。训练对每条真实序列均匀选四种采样布局之一，从该序列本身取空间监督，保留 10% 空间条件清空。每项任务的训练布局和生成布局共用 `contracts.task_mask`。

修正版空间引导选择固定公开原仓库的逐点 L2、方差下界 0.01、采样密度缩放和完整迭代日程。训练在加噪后进行引导，按各条样本时间步在低噪声区间使用 100 次，其余 20 次。采样在反向均值上进行引导，最后 10 步各 500 次，其余各 10 次。支持显式掩码、变长和混合训练时间步。这里采用的是公开源码的 L2，v2 论文描述为 MSE。 [固定公开实现](https://github.com/GenRobo/DreamControl/blob/ec28b508313d6837de52c68247fbc8cf81fe4935/TrajGen/diffusion/gaussian_diffusion.py#L423) · [v2 论文采样](https://arxiv.org/html/2604.00202v1#S3.SS2.SSS3)

生成先从静态 22 点站姿建立与训练编码相同的 canonical 坐标，减去该姿态的最低关键点高度和初始骨盆平面位置，再对齐朝向。按相同变换处理所有世界目标并保存逆变换。物理地板仍由场景的 `ground_z` 定义。Pick 保留项目桌面和物体位置采样，补充右肘目标、固定右手语义、40–50 帧接触采样、右手闭合时序和腕朝向。Pour、双手搬箱与推拉沿用既有世界目标和时序。控制目标与物体计划均为生成条件。

## 配置与调用

使用既有服务器 Python、PyTorch 和冻结资产。新 JSON 继承归档 baseline config，最小改动如下。路径以实际授权的服务器项目目录为准。

```json
{
  "baseline_config": "/absolute/project/stage2_prior29_20261103/config.json",
  "stage_dir": "/absolute/project/repair_prior29_v2_20261105",
  "run_name": "grab_omni29_mdm_floor_l2_guided_v2",
  "max_sample_batch_size": 4
}
```

`max_sample_batch_size` 默认为 4，静态配置允许 1–32。容量见证实际通过后再提高调用批量。采样在训练设备上运行，或在启动前配置 `sample_device`。采样分配器份额 `sample_gpu_memory_fraction` 默认为 0.05，受既有 0.10 硬上限和显存余量检查。全部覆盖项纳入保存的配置，所以训练和采样应使用同一份配置文件。远端原 CMDM SHA-256 必须匹配固定公开文件 `aa7e1d5e1e297a0e27b31db64f1d12a934f5de9f8e24572afb0fbc501fc30a0f`。

```sh
"$PY" -B "$SRC/cpu_contract_test.py"
"$PY" -B -u "$SRC/train.py" --config "$CFG" --mode preflight
"$PY" -B -u "$SRC/train.py" --config "$CFG" --mode smoke
"$PY" -B -u "$SRC/train.py" --config "$CFG" --mode train
"$PY" -B -u "$SRC/sample.py" --config "$CFG" --checkpoint "$FINAL" --task pick \
  --count 1 --start-index 0 --batch-size 1 --output-dir "$STAGE/candidate_witness/pick"
```

`smoke` 使用八次真实更新见证模型与优化器，检查混合长度 padding 不改变有效帧，并分别走 t=50、t=0 的真实 CFG 反向步骤。完整检查点在内存序列化并用 CPU `weights_only` 逐项读回，核对模型、Adam 一二阶矩与步数、RNG、配置、数据来源和游标。正式保存时对实际文件做同样校验，写 `step_*.validation.json`，记录 SHA 和 Adam 更新分布。保存后读回失败会留下文件及失败记录，续训和采样仅接收带有效验证收据的检查点。

末行 `sample.py` 使用正式最终权重完成一次完整 1,000 步采样，才可作为后优化入口见证。默认从最终检查点同目录的完成状态和读回收据验证 SHA。若从冷存路径读取相同权重，请随检查点恢复相邻验证收据并显式传 `--checkpoint-sha256`。采样器对四类任务都支持独立的任务名、批大小、种子和起始候选 ID。候选数等于尝试次数，合格后优化参考数由下游准入决定。

## NPZ 与坐标接口

| 字段 | 尺寸或解释 |
| --- | --- |
| `features` | `(1,T,263)`，反标准化的 263D |
| `canonical_points` | `(1,T,22,3)`，模型空间的解码点 |
| `target_points` 和 `target_points_world` | `(T,22,3)`，同一组 canonical 与世界控制点 |
| `target_mask` | `(T,22)`，布尔 |
| `desired_wrist_quat_wxyz` | `(T,2,4)`，世界系，左右顺序 |
| `planned_hand_closed` | `(T,2)`，左右顺序，0/1 |
| `desired_object_positions` | `(T,3)`，世界系的物体计划 |
| `metadata_utf8` | JSON，含候选、任务、`grasp_frame`、`grasp_time_s`、来源哈希与坐标变换 |

行向量约定为 `world = canonical @ canonical_to_world_rotation.T + canonical_to_world_translation`。逆变换用对应的 `world_to_canonical_*`。样本同时保存世界目标字节哈希和可逆验证需要的两个变换。`coordinate_provenance` 保存 `training_floor_canonical_v2`、物理地板 z、站姿高度基准、关键点与 URDF 哈希、关节顺序及校准哈希。下游从样本读取变换和 grasp frame。

## 验证状态

本机 Python 3.12 NumPy 运行了四个检查，通过 3.10 语法解析、隔离运行配置、任务掩码、迭代日程和坐标往返。坐标检查与既有训练编码在 1e-12 容差内吻合。本机跳过三个依赖 PyTorch 或 SciPy 的检查，包括完整检查点读回、CPU 自动求导和四元数插值。在已绑定服务器 Python 可运行同一 CPU 脚本覆盖这三项。这里尚无新 GPU 容量、训练、完整采样、后优化准入或五指标的结果。


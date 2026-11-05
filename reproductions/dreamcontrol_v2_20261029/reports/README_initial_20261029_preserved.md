# DreamControl-v2：仿真复现工作区

分阶段执行计划与验收条件见 [PLAN.md](PLAN.md)；最近状态见 [STATUS.md](STATUS.md)。

本目录是独立的复现工作区，目标为论文前三阶段：统一 G1 运动数据、机器人空间条件扩散先验、仿真 PPO；不包含真机部署。当前仅完成**数据检查和实验性单样本预处理**，没有训练 DreamControl-v2 扩散模型或 PPO 策略，也没有复现论文性能。

## 已核查的来源

- 论文：https://arxiv.org/html/2604.00202 。作者 [DreamControl-v2 仓库](https://github.com/GenRobo/DreamControl-v2) 当前为项目网页；不是已确认可用的 v2 训练实现。
- 服务器现有 DreamControl 一代项目：`/data/code/ouyanghao/oyh/conditional-human-motion-generation`，上游记录为 `GenRobo/DreamControl` 的 `ec28b508313d6837de52c68247fbc8cf81fe4935`。服务器工作树有既存未提交更改，不能重置或直接覆盖。
- 本地 `assets/g1_27dof.urdf` 是从该服务器已有上游 G1 资产取得的拷贝，SHA-256：`652c8e1ed1c41a2ed843ac16d7090b1faffd53af127b246c725e800d80d4b27a`。资产遵循上游项目及模型条款。
- 已找到 `reactivebfm-reproduction-20261021/data/amass_g1_36dim/full_train.pkl`：按其项目文档为 60 Hz、Z-up、XYZ + XYZW + 29 关节；其训练／验证／测试划分与 v2 论文数据划分不应默认相同。
- 另找到 `/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass` 约 14 GB 的已有 AMASS 相关材料，包括多个原始人体动作子集和配对记录；这些不是已经构建好的 v2 G1 数据集。原始数据覆盖率、授权与样本到文本的对应关系仍需核验。
- 2026-10-29 对 `/data` 和 `/home/ouyanghao` 最深 5 层的名称进行只读搜索，未发现名称匹配 GRAB、Nymeria、OmniRetarget 的数据目录；这不是对更深层或其他挂载点的穷尽证明。原 DreamControl 的 HumanML3D 数据目录仅找到 5 个 `new_joint_vecs` 和 5 个文本文件；并不能据此否定其他项目有数据。

## 当前实现

`src/audit_g1_corpus.py` 检查既有 36 维语料的结构、划分、四元数、文本和两个将被锁定的腰部关节。**它会加载 pickle，只能对可信数据运行。**

`src/g1_fk.py` 读取 G1 URDF，按论文附录表 VII 的 22 个链接输出 3D 关键点；`src/motion_bridge.py` 将符合明确腰角容差的 29 关节样本降采样到 20 Hz、映射为 27 关节并计算关键点。如果腰部角度超出容差，代码直接拒绝：这不是重定向优化器。

`src/feature263.py` 将 Z-up 关键点变换到右手系 Y-up、进行初始方向归一化，调用现有 DreamControl HumanML 特征提取器输出 263 维序列。`src/prepare_clip.py` 将上述步骤组合为单样本命令行，保存数据及来源哈希，拒绝覆盖已有输出。这里没有使用 `process_file` 中的人体骨架统一缩放；足部阈值和地面估计为显式复现选择，**尚未同 v2 作者实现核验**。

`src/actor_observation.py` 是独立的 PyTorch 点目标 actor 观测原型，以 Isaac Lab 的 WXYZ 根部四元数将世界目标转换到机器人坐标系；其函数输入没有参考轨迹。当前未接入 Isaac Lab，也未包含物体姿态与各任务的其他传感量，不能当成已完成的八任务 PPO actor。

`src/waist_lock_ik.py` 是现有 29 自由度 G1 运动到 27 自由度的实验性拟合：从旧机器人取 12 个局部关键点，在新的锁腰机器人上用可微运动学优化关节角，显式控制关节限位、原动作偏移及速度正则。它不是论文的人体到机器人的重定向实现，也没有完成足部接触和动力学验收；所有优化超参数均为本项目探索设置。

单条可信 `motion.npy` 的输入格式：`(T,36)`，`[0:3]` 世界坐标、`[3:7]` XYZW、`[7:36]` ReactiveBFM README 的 29 关节顺序。该代码仅支持整数倍降采样；如果 29 自由度腰部确实运动，必须先进行 27 自由度重优化，不能将其输入用于论文训练。

在已有服务器环境中，进入原 DreamControl 项目目录后可用：

```bash
PYTHONPATH=third_party/DreamControl/TrajGen:envs/dreamcontrol-r3/site-packages \
  /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python \
  /path/to/dreamcontrol-v2/src/prepare_clip.py \
  --motion /path/to/trusted_raw_36d.npy \
  --urdf third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf \
  --output /path/to/new_output.npz --source-fps 60 \
  --waist-tolerance-rad 0.01 --foot-velocity-squared-threshold 0.002
```

上面的 `0.01` 和 `0.002` 是检查用的近似设置，不是经确认的 v2 论文参数。输出特征有 `T-1` 帧，因为特征包括相邻帧速度；输出并非训练数据管线的完整交付。运行条件和数据许可需要用户环境自行满足。

## 实测证据

- [语料审计](reports/amass_g1_corpus_audit.json)：11,424 条结构有效；11,418 条至少一帧腰部 roll/pitch 超出 0.01 rad。四元数最大模长误差约 `5.96e-8`。这些数字不证明语义质量、重定向质量或原始帧率无误。
- [FK 对照](reports/fk_reference_witness.json)：4 组 G1 关节角、22 点，与现有 `pytorch_kinematics` 的最大位置差 `1.14e-8 m` 以内；并有独立的简单几何单元测试。
- [6 条真实数据桥接](reports/real_amass_bridge_witness.json)：以 0.01 rad 腰角近似容差检查；其余 11,418 条不满足该条件。
- [编码与恢复检查](reports/real_amass_263_witness.json)：6 条均输出有限的 `(T-1,263)`，通过原 HumanML `recover_from_ric` 恢复关键点，最大误差约 `6.8e-8 m`。这只是位置特征自洽检查，不验证 6D 旋转、足部接触、动作真实性或作者 v2 等价性。
- [锁腰拟合 CPU 片段试验](reports/waist_lock_ik_witness.json)：一个 AMASS 样本的 6 帧中，12 个关键点平均偏差从约 15.10 mm 降至 12.00 mm；最大偏差仍约 50.01 mm。随后产生 5 帧 263 维特征。仅为初步测试，不代表整段轨迹或论文质量通过。
- `tests/` 包含有限值、划分重叠、锁腰拒绝、基本运动学和坐标归一化的本地检查。另有 `fk_external_witness.py`、`feature_external_witness.py` 记录服务器上已有库对照的调用内容。
- 服务器 CPU PyTorch 下 actor 点目标转换的两项检查通过，覆盖根坐标系旋转、全局平移不变性和零四元数拒绝；这不是 Isaac 环境中的训练或观测依赖测试。

## 下一步和未完成项

1. 复用或实现 29→27 自由度受约束重优化，并对关键点、脚接触、关节限位、坐标和运动失真进行检查；确认 AMASS 数据的出处、许可、真实采样率、切分方式。完成后才可建立可训练的 263D 语料及仅训练集统计量。
2. 获取并核验 GRAB、Nymeria、OmniRetarget，补齐其预处理、Xsens→SMPL 转换与 Full-Mix 分组；不能将 AMASS 子集称为 Full-Mix。
3. 以 MDM 预训练权重及 OmniControl 训练代码实现训练和 Realism Guidance，设置可审计的训练配置、检查点与数据来源。论文表 V 使用约 102k 次迭代；没有实际训练结果之前不得称为完成。
4. 实现空间梯度引导、任务采样与作者附录中的后优化和筛选；对每项任务记录失败、保留率、接触时序和指标。
5. 对 Isaac Lab actor 配置做参考依赖审计并进行适配。旧版 Drawer `PolicyCfg` 显式包含 `target_ref_*`；甚至其 `*_real` 版本中的 `hand_state_target*` 函数仍查询 `motion_lib`。v2 的 actor 不能只靠删除 `target_ref` 字段来证明无参考依赖，应使用真实任务目标观测，并在运行时改变参考而固定物理状态来验证 actor 输出不变。
6. 多任务 PPO、论文表 I–III 的生成与仿真指标、必要消融仍未运行。当前两张服务器 GPU 有其他进程，`/data` 只剩约 21 GB；不占用或干扰现有工作，不以小测试充当正式复现。

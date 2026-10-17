# 课程学习（Curriculum Learning）使用说明

本文档介绍 Tracking 任务中新增的"基于难度等级的课程学习"功能，支持从简单动作逐步过渡到复杂动作的渐进式训练。

## 1. 功能概述

课程学习通过以下机制实现渐进式训练：

1. **难度分级**：根据 CSV 文件中的难度标注（1-10级），将动作文件按难度分组
2. **分阶段解锁**：训练从最简单的动作开始，达到性能阈值后逐步解锁更高难度
3. **定期评估**：每隔固定迭代次数，对当前难度等级的动作进行专门评估
4. **阈值判定**：评估指标（mpjpe、mpjae）达标后自动解锁下一等级

## 2. 数据准备

### 2.1 难度标注 CSV 文件

CSV 文件格式（`filtered_texts_new.csv`）：

| 列名 | 说明 | 示例 |
|------|------|------|
| file | 动作文件名（.txt 格式） | 000000.txt |
| rating | 难度等级（1-10） | 6 |
| reason | 难度评估原因 | 核心动作为踢腿，属于中等难度 |
| action | 动作描述 | a man kicks something... |

**注意事项**：
- CSV 中的 `000000.txt` 会自动映射到 `000000.npz`
- 找不到对应 npz 文件的条目会被跳过并记录
- 难度 > 10 的动作（如需要特殊地形）默认被排除

### 2.2 mocap 数据

`/ssd/humanoid/dataset/mocap_npz` 目录下的所有文件会被自动标记为难度 5，在解锁到第 5 级时加入训练。

## 3. 配置说明

### 3.1 MotionCommandCfg 新增字段

```yaml
commands:
  motion:
    # 启用课程学习
    curriculum_enabled: true
    
    # 难度标注 CSV 文件路径
    curriculum_csv_path: "/path/to/filtered_texts_new.csv"
    
    # mocap 数据目录（所有文件默认难度 5）
    curriculum_mocap_dir: "/ssd/humanoid/dataset/mocap_npz"
    
    # 每个难度等级训练的迭代次数
    curriculum_iteration_per_level: 6000
    
    # 评估通过阈值
    curriculum_mpjpe_threshold: 0.12
    curriculum_mpjae_threshold: 0.06
    
    # 新解锁等级的采样权重（相对于旧等级的 1.0）
    curriculum_new_level_weight: 2.0
    
    # 评估时每个文件运行的次数
    curriculum_eval_runs: 5
    
    # 是否使用子进程进行精确评估（推荐开启）
    curriculum_use_subprocess_eval: true
    
    # 子进程评估时使用的环境数量
    curriculum_eval_num_envs: 1000
    
    # 最大难度等级（排除更高难度的文件）
    curriculum_max_difficulty: 10
    
    # 防遗忘：每个已解锁等级的最低采样比例
    # 例如 0.1 表示每个等级至少占总采样的 10%
    curriculum_min_level_sampling_ratio: 0.1
    
    # ========== 文件渐进引入配置（解决新level学习效率低的问题） ==========
    # 新等级的文件逐步引入，让策略有时间充分学习每批新动作
    curriculum_gradual_file_intro_enabled: true
    # 渐进引入持续迭代数
    curriculum_gradual_file_intro_iters: 5000
    # 起始引入比例（10%）
    curriculum_gradual_file_intro_start_ratio: 0.1
    
    # ========== Level 变化时参数提升配置 ==========
    # 当升级到新 level 时，调整训练参数以加速学习新动作
    # 参数会在 curriculum_gradual_file_intro_iters 期间逐渐恢复到初始值
    curriculum_lr_reset_enabled: true         # 启用参数调整（升级时 lr 恢复到初始值）
    curriculum_kl_boost_ratio: 1.5            # desired_kl 提升倍数（0.01 -> 0.015）
    curriculum_entropy_boost_ratio: 2.0       # entropy_coef 提升倍数（0.005 -> 0.01）
```

### 3.2 完整配置示例

参考 `g1_training_curriculum.yaml`：

```yaml
runner:
  num_steps_per_env: 24
  max_iterations: 100000
  experiment_name: "g1_flat_curriculum"

commands:
  motion:
    curriculum_enabled: true
    curriculum_csv_path: "/ssd/bjc/bm_mlp/source/whole_body_tracking/whole_body_tracking/utils/filtered_texts_new.csv"
    curriculum_mocap_dir: "/ssd/humanoid/dataset/mocap_npz"
    curriculum_iteration_per_level: 6000
    curriculum_mpjpe_threshold: 0.12
    curriculum_mpjae_threshold: 0.06
    curriculum_new_level_weight: 3.0
    curriculum_eval_runs: 10
```

## 4. 训练流程

### 4.1 启动命令

```bash
python scripts/rsl_rl/train.py --task Tracking-Flat-G1-v0 \
  --motion_dir /ssd/humanoid/dataset/humanml_gmr/npz \
  --g1_training_cfg source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/cfg/g1_training_curriculum.yaml \
  --headless --logger wandb
```

### 4.2 训练过程

```
┌─────────────────────────────────────────────────────────────────┐
│  开始训练（Level 1: 644 个简单动作）                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  训练 6000 次迭代                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  暂停训练，运行评估                                              │
│  - 对 Level 1 的所有文件（排除 frozen）进行评估                   │
│  - 每个文件运行 10 次，取平均                                     │
│  - 计算 mpjpe 和 mpjae                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌─────────────────┐             ┌─────────────────┐
    │  通过评估        │             │  未通过评估      │
    │  mpjpe < 0.12   │             │                 │
    │  mpjae < 0.06   │             │                 │
    └─────────────────┘             └─────────────────┘
              │                               │
              ▼                               │
    ┌─────────────────┐                       │
    │  解锁 Level 2    │                       │
    │  (+ 5329 个文件) │                       │
    └─────────────────┘                       │
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  继续训练...                                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 难度等级分布

基于当前 CSV 文件的统计：

| 等级 | 文件数 | 累计文件数 | 说明 |
|------|--------|-----------|------|
| 1 | 644 | 644 | 最简单（静止手势等） |
| 2 | 5,329 | 5,973 | 简单（慢走、搬运等） |
| 3 | 2,764 | 8,737 | 较简单 |
| 4 | 1,095 | 9,832 | 中等偏低 |
| 5 | 2,050 | 11,882 | 中等（含 mocap 数据） |
| 6 | 657 | 12,539 | 中等偏高（踢腿、跳跃等） |
| 7 | 445 | 12,984 | 较难（拳击组合等） |
| 8 | 168 | 13,152 | 难（倒立等） |
| 9 | 32 | 13,184 | 很难 |
| 10 | 11 | 13,195 | 最难 |

### 4.4 防遗忘机制：分层采样

由于使用 MLP 架构，存在**灾难性遗忘**的风险：学习新动作时可能遗忘旧动作。

本系统通过**分层采样**策略实现防遗忘：

#### 采样策略（三层机制）

1. **旧等级**：按**文件数比例**分配配额（不是固定 10%）
2. **新等级**：获得 `new_level_weight` 倍加权（默认 2x）
3. **最低保证**：每个等级至少 `min_level_sampling_ratio`（默认 5%）

#### 实际效果示例

**Max Level = 5**（new_level_weight=2.0, min_ratio=0.05）：

| Level | 文件数 | 文件比例 | 实际采样比例 | 说明 |
|-------|--------|---------|-------------|------|
| 1 | 644 | 5.42% | 5.00% | 最低保证 |
| 2 | 5,329 | 44.85% | **38.10%** | 按文件数比例 |
| 3 | 2,764 | 23.26% | 19.76% | 按文件数比例 |
| 4 | 1,095 | 9.22% | 7.83% | 按文件数比例 |
| 5 | 2,050 | 17.25% | **29.31%** | 新等级 2x 加权 |

**Max Level = 7**（new_level_weight=2.0, min_ratio=0.05）：

| Level | 文件数 | 文件比例 | 实际采样比例 | 说明 |
|-------|--------|---------|-------------|------|
| 2 | 5,329 | 41.04% | **39.55%** | 文件多，配额多 |
| 3 | 2,764 | 21.29% | 20.51% | 按比例 |
| 7 | 445 | 3.43% | **6.60%** | 新等级 2x 加权 |

**核心优势**：
- Level 2 有 5329 个文件，获得 ~40% 配额（而非固定 10%）
- 避免大量文件被过度压缩
- 新等级仍获得加权，保证训练重点
- 文件少的等级（如 Level 1）有最低保证

### 4.5 文件渐进引入机制

**问题**：即使有渐进式权重过渡，新level一次性引入大量文件（如Level 2有5329个新文件）仍会导致学习效率低下，因为每个文件的采样次数太少。

**解决方案**：文件渐进引入。新等级解锁后，不是一次性引入所有文件，而是从 `start_ratio`（默认10%）逐步增加到100%。

#### 配置参数

```yaml
commands:
  motion:
    curriculum_gradual_file_intro_enabled: true  # 启用文件渐进引入
    curriculum_gradual_file_intro_iters: 5000    # 渐进引入持续迭代数
    curriculum_gradual_file_intro_start_ratio: 0.1  # 起始引入比例（10%）
```

#### 效果示例

假设 Level 2 有 5329 个文件，在 iter=5000 解锁，`intro_iters=4000`，`start_ratio=0.1`：

| 迭代 | 引入进度 | 活跃文件数 | 每文件采样密度 |
|------|---------|-----------|--------------|
| 5000 | 0% | 533 (10%) | 高（充分学习）|
| 6000 | 25% | 1732 (32.5%) | 中 |
| 7000 | 50% | 2931 (55%) | 中 |
| 8000 | 75% | 4130 (77.5%) | 较低 |
| 9000 | 100% | 5329 (100%) | 正常 |

**核心优势**：
- 每批新文件有足够的采样次数来学习
- 避免"样本稀释"导致的学习效率低下
- 策略可以逐步扩展能力范围

**推荐配置**：
```yaml
curriculum_gradual_file_intro_enabled: true
curriculum_gradual_file_intro_iters: 4000
curriculum_gradual_file_intro_start_ratio: 0.1
```

### 4.6 Level 变化时训练参数提升

**问题**：升级到新 level 时，策略需要快速适应新动作，但此时学习率可能已经被 adaptive schedule 降低，导致学习缓慢。

**解决方案**：Level 升级时自动调整训练参数，然后在文件渐进引入期间逐渐恢复到初始值。

#### 参数调整机制

| 参数 | 升级时变化 | 恢复方式 |
|------|-----------|---------|
| `learning_rate` | 重置到初始值 | adaptive 自动调整 |
| `desired_kl` | 提升 1.5x | 线性恢复到初始值 |
| `entropy_coef` | 提升 2.0x | 线性恢复到初始值 |

**恢复周期**：与 `curriculum_gradual_file_intro_iters` 同步

#### 配置参数

```yaml
commands:
  motion:
    # ========== Level 变化时参数提升配置 ==========
    # 当升级到新 level 时，调整训练参数以加速学习新动作
    # 参数会在 curriculum_gradual_file_intro_iters 期间逐渐恢复到初始值
    curriculum_lr_reset_enabled: true         # 启用参数调整（升级时 lr 恢复到初始值）
    curriculum_kl_boost_ratio: 1.5            # desired_kl 提升倍数（0.01 -> 0.015）
    curriculum_entropy_boost_ratio: 2.0       # entropy_coef 提升倍数（0.005 -> 0.01）
```

#### 效果示例

假设在 iter=5000 升级到 Level 2，`curriculum_gradual_file_intro_iters=2000`：

| 迭代 | learning_rate | desired_kl | entropy_coef | 说明 |
|------|---------------|------------|--------------|------|
| 4999 | 0.0005 | 0.01 | 0.005 | Level 1 稳定（lr 被 adaptive 降低）|
| **5000** | **0.001** | **0.015** | **0.01** | Level 2 升级！参数提升 |
| 5500 | adaptive | 0.0138 | 0.0088 | 恢复 25% |
| 6000 | adaptive | 0.0125 | 0.0075 | 恢复 50% |
| 6500 | adaptive | 0.0113 | 0.0063 | 恢复 75% |
| **7000** | adaptive | **0.01** | **0.005** | 恢复完成，回到初始值 |

#### 设计原理

1. **learning_rate 重置**：让策略以"全速"开始学习新动作，然后 adaptive 机制根据 KL 散度自动找到合适的学习率

2. **desired_kl 提升**：允许更大的策略变化幅度
   - 正常训练时 KL 太大会触发 lr 降低
   - 新 level 时临时提高 desired_kl，容忍更大的策略变化
   - 随着文件逐步引入，恢复到保守值以稳定训练

3. **entropy_coef 提升**：增加探索程度
   - 新动作需要更多探索来发现正确的行为
   - 逐渐恢复到较低值，让策略从探索转向利用

#### 训练日志输出

```
[Curriculum Param Boost] Level 1 -> 2
[Curriculum Param Boost] learning_rate: 0.000500 -> 0.001000 (reset to initial)
[Curriculum Param Boost] desired_kl:    0.0100 -> 0.0150 (x1.5)
[Curriculum Param Boost] entropy_coef:  0.0050 -> 0.0100 (x2.0)
[Curriculum Param Boost] Will decay back to initial over 2000 iterations
```

#### 与其他机制的协同

```
Level 升级触发
    │
    ├──► 文件渐进引入：10% -> 100% 新文件
    │
    ├──► 参数提升：lr 重置，kl/entropy 提升
    │       └──► 逐渐恢复到初始值
    │
    └──► 采样权重：新 level 获得 curriculum_new_level_weight 倍权重

三个机制同步进行，在 curriculum_gradual_file_intro_iters 内完成过渡
```

#### 等级内采样：Error-Based 自适应

在每个等级内部，仍然使用 error-based 自适应采样：

```
最终采样概率 = 等级配额 × (等级内 error-based 分布)

例如 Level 2 (38% 配额，5329 文件):
- error 高的文件在 Level 2 内被更多采样
- 整个 Level 2 总采样概率约为 38%
- 既保证文件数多的等级有足够配额
- 又在等级内部让难的文件被多练习
```

这确保即使训练到最高难度，所有已学动作仍会按合理比例被训练。

## 5. 评估指标

### 5.1 mpjpe（Mean Per-Joint Position Error）

身体各关节位置误差的平均值：

```python
mpjpe = torch.norm(body_pos_w - robot_body_pos_w, dim=-1).mean()
```

阈值：`< 0.12 米`

### 5.2 mpjae（Mean Per-Joint Angle Error）

各关节角度误差的平均值：

```python
mpjae = torch.abs(joint_pos - robot_joint_pos).mean()
```

阈值：`< 0.06 弧度`

## 6. 监控指标

训练过程中会记录以下指标到 wandb：

| 指标名 | 说明 |
|--------|------|
| `Curriculum/level` | 当前解锁的最高难度等级 |
| `Curriculum/eval_mpjpe` | 评估时的 mpjpe |
| `Curriculum/eval_mpjae` | 评估时的 mpjae |
| `Curriculum/lr_boosted` | Level 升级时重置后的学习率 |
| `Curriculum/kl_boosted` | Level 升级时提升后的 desired_kl |
| `Curriculum/entropy_boosted` | Level 升级时提升后的 entropy_coef |
| `Curriculum/sampling_ratio_level_N` | 各等级的实际采样比例 |
| `Curriculum/num_unlocked_files` | 当前已解锁的总文件数 |

## 7. 实现细节

### 7.1 文件结构

```
source/whole_body_tracking/whole_body_tracking/
├── utils/
│   ├── curriculum_manager.py    # 课程学习管理器
│   └── my_on_policy_runner.py   # 扩展的训练运行器
├── tasks/tracking/
│   ├── mdp/
│   │   └── commands.py          # MotionCommand 扩展
│   └── config/g1/cfg/
│       └── g1_training_curriculum.yaml  # 示例配置
└── docs/
    └── curriculum_learning.md   # 本文档
```

### 7.2 关键类

1. **CurriculumManager**：管理难度等级和文件分组
2. **MotionCommand**：扩展了课程学习相关方法
3. **MotionOnPolicyRunner**：在训练循环中插入评估检查点
4. **CurriculumEvaluator**：管理评估任务的并行调度
5. **MotionTermStateSnapshot**：保存/恢复 motion term 状态

### 7.3 评估过程（子进程精确评估）

评估通过**启动独立的 Isaac Lab 子进程**进行，确保精确计算 mpjpe 和 mpjae：

1. 在每个 `curriculum_iteration_per_level` 迭代检查点触发评估
2. 保存当前模型检查点和评估文件列表
3. 启动 `curriculum_eval.py` 子进程：
   - 创建新的 Isaac Lab 环境
   - 加载模型检查点
   - 对当前等级的所有文件进行完整评估
   - 每个文件运行 `eval_runs` 次，计算平均 mpjpe 和 mpjae
4. 读取子进程输出的评估结果
5. 如果 mpjpe 和 mpjae 低于阈值，解锁下一等级

**评估配置**：

```yaml
commands:
  motion:
    curriculum_mpjpe_threshold: 0.12        # mpjpe 阈值（米）
    curriculum_mpjae_threshold: 0.06        # mpjae 阈值（弧度）
    curriculum_use_subprocess_eval: true    # 是否使用子进程评估
    curriculum_eval_runs: 5                 # 每个文件评估次数
    curriculum_eval_num_envs: 1000            # 子进程环境数量
```

**优点**：
- 精确计算 mpjpe 和 mpjae（完整 clip 评估）
- 只评估当前等级的文件
- 避免 inference_mode tensor 冲突（独立进程）
- 评估结果可复现

**注意事项**：
- 子进程评估需要额外时间（取决于文件数量和 eval_runs）
- 如果子进程失败，会回退到 EMA 评估
- 可通过 `curriculum_use_subprocess_eval: false` 禁用子进程评估

## 8. 向后兼容

- 默认 `curriculum_enabled: false`，不影响现有训练流程
- 所有原有配置和脚本保持兼容
- 可以随时在配置中启用/禁用课程学习

## 9. 常见问题

### Q: 如果某个等级一直无法通过怎么办？

A: 训练会继续在当前等级进行，每隔 `curriculum_iteration_per_level` 次迭代会重新评估。可以考虑：
- 降低阈值
- 增加每个等级的训练迭代次数
- 检查是否有太难的动作被错误标注

### Q: 如何跳过某些等级？

A: 目前不支持跳级，但可以通过调低阈值快速通过简单等级。

### Q: 能否从中间等级开始？

A: 目前从 Level 1 开始。如果需要从中间等级开始，可以手动修改 `CurriculumManager.max_level` 初始值。

### Q: 评估时间太长怎么办？

A: 可以减少 `curriculum_eval_runs` 的值，或者增加并行环境数量。


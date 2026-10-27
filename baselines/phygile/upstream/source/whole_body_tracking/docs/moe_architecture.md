# MoE (Mixture-of-Experts) Actor 架构

## 概述

将原有的 MLP Actor 替换为 Mixture-of-Experts 架构，采用 **motion-conditioned soft top-k routing**：

- Router 输入为 **z_t**（未来动作 embedding），而非完整 observation
- 每 M 步更新候选 expert 集合，步间用 `softmax(logits/T)` 平滑混合
- 训练分两大阶段：
  - **Phase A（课程约束路由）**：按 curriculum level 逐步解锁 expert，router 通过 CE loss 学习
  - **Phase B（动态新增 expert）**：对无标注 motion，先用现有 experts 组合覆盖，覆盖不了时新增 expert

Critic 保持单一 MLP 不变。FutureCommandProcessor 与 MLP 版本完全一致。没有独立的 Encoder MLP（`encoder_hidden_dims: null`）。

## 架构

注意：`encoder_hidden_dims: null`，不存在独立的 Encoder MLP 层。
FutureCommandProcessor 的输出直接拼接后送入 Expert MLP。

```
Observation (o_t)
    |
    v
FutureCommandProcessor (actor/critic shared)
    |                               \
    v                                v
processed_obs                      z_t (future conv latent, 128-dim)
= [raw_cmd, z_t_latent, rest]       |
    |                                v
    |                          Gate Network (MLP [256])
    |                                |
    |                                v
    |                          gate_logits (num_experts)
    |                                |
    |                          Curriculum Mask: logits[inactive] = -inf
    |                                |
    |                          Every M steps: K = topk(logits, k=3)
    |                                |
    |                          Every step: p = softmax(logits[K] / T)
    |                                |
    +----> Expert_K[0](processed_obs) --> out_0    weight p_0
    +----> Expert_K[1](processed_obs) --> out_1    weight p_1
    +----> Expert_K[2](processed_obs) --> out_2    weight p_2
                                            |
                                   a = sum(p_i * out_i)
                                            |
                                            v
                                      action_mean
```

## 路由机制：Motion-Conditioned Soft Top-K

核心改动：将 hard top-k 替换为 soft top-k + 低频候选更新。

### 与旧 Hard Top-K 的区别

| 特性 | Hard Top-K (旧) | Soft Top-K (新) |
|------|-----------------|-----------------|
| 候选更新频率 | 每步 | 每 M 步 (M=10) |
| 权重计算 | topk(softmax(logits)) | softmax(logits[K] / T) |
| Router 输入 | encoded_obs | z_t (未来动作 latent) |
| 温度控制 | 无 | T=0.5 (更集中) |
| Logits 平滑 | 无 | EMA alpha=0.8 (rollout) |
| Hard routing | 80% 全部硬路由 | 80% 仅新 level 硬路由 |

### Rollout 模式（推理/收集数据）

1. `z_t = FutureCommandProcessor.forward_with_latent(cmd).latent`
2. `logits = Gate(z_t)`
3. `logits_smooth = alpha * logits + (1-alpha) * prev_logits`（EMA）
4. 每 M 步：`K = topk(logits, k=3)` 更新候选集
5. 每步：`p = softmax(logits[K] / T)` 计算 soft 权重
6. 只 forward 候选 experts：`a = sum(p_i * E_K[i](encoded))`

### Training 模式（PPO 更新）

- 每个 mini-batch 重新选择候选（无低频约束）
- CE loss 或 load-balance loss 训练 router
- 梯度通过 softmax 和 expert outputs 正常回传

### 新 Level 样本 Hard Routing

为确保每个 expert 学到对应 level 的特征，新 level 样本（`level == current_max_level`）采用 hard routing：

| Level | Hard 部分 (80%) | Soft 部分 (20%) |
|-------|----------------|-----------------|
| 1 | expert 0 only | - (只有1个专家) |
| 2 | expert 1 only | gate 选 top-2 from {0,1} |
| 3 | expert 2 only | gate 选 top-3 from {0,1,2} |
| 4+ | expert (level-1) only | gate 选 top-3 from {0..level-1} |

训练时逻辑（仅在梯度开启时执行）：
```python
# 识别新 level 样本
is_new_level = (level_labels == current_max_level)
# 随机选择 80% 使用 hard routing
hard_mask = is_new_level & (torch.rand(B) < 0.8)
# Hard routing: 强制只用对应 expert
if hard_mask.any():
    action_mean[hard_mask] = experts[level-1](encoded[hard_mask])
```

这样设计的好处：
- **Hard 部分 (80%)**：确保新 expert 有足够梯度学习对应 level 的特征
- **Soft 部分 (20%)**：让 gate 学习如何组合专家，为推理时的动态路由做准备

### Expert Warm Start

升级到新 level 时，自动将上一个 expert 的参数复制给新 expert：

```
Level 1 → Level 2: expert_1.load_state_dict(expert_0.state_dict())
Level 2 → Level 3: expert_2.load_state_dict(expert_1.state_dict())
...
```

好处：
- 新 expert 不从零开始，已有基础运动能力
- 只需 fine-tune 学习新 level 的特定特征
- 训练速度显著加快

## Phase A：课程约束路由

### A1. 课程数据组织

每个 level L 有数据集 D_L。训练采样分布由 CurriculumManager 管理：
- 新 level 权重 `new_level_weight=10.0`
- 旧 level 最低采样比例 `min_level_sampling_ratio=0.01`
- **Difficulty-Weighted 采样**：`level_difficulty_weights: [1, 1, 1.2, 1.5, 2, 3, 4, 6, 8, 10]`，高 level 文件少但难度大，给予更高 per-file 采样密度
- 文件渐进引入（gradual file intro）

### A1b. 课程升级策略

升级决策三条件 OR（满足任一即升级）：

1. **指标达标**：mpjpe < threshold OR mpjae < threshold
2. **Reward 收敛**：最近 W=20 次 reward 记录前后半段改善率 < 2%，且已训练 >= 3000 iters
3. **Per-level 硬超时**：`[5000, 5000, 6000, 8000, 10000, 12000, 15000, 20000, 20000, 20000]`

低 level（lv1-2）通常指标直接通过；中 level（lv3-7）指标不够但 reward 收敛后提前升级；高 level（lv8-10）最多等 timeout 但通常收敛后就走。

### A2. 课程约束的 Router 训练

在 level L 时：
- 允许使用的 expert 集合：`A_L = {0, 1, ..., L-1}`
- Router logits 做 mask：`logits[not in A_L] = -inf`
- 然后 soft top-k (k=3) 选择候选

CE loss 训练 router：`CE(gate_logits, level_label)` 让 router 学会根据 z_t 选择对应的 expert。

### A3. 优化目标

- **主损失**：PPO surrogate loss + value loss（用混合 action 交互产生）
- **辅助损失**：`routing_loss_coef * CE(gate_logits, level_label)`
- **路由稳定性**：
  - 候选集低频更新：每 M=10 步（@50Hz 约 0.2s）
  - Logits EMA：`alpha=0.8`（可选）
  - 温度 T=0.5：让路由更集中

### A4. Phase A 结束时得到

- 一组已专精的 experts（0..K-1）
- 一个能用 z_t 选择/组合这些 experts 的 router
- 课程采样策略确保旧 expert 不遗忘

## Phase B：动态新增 Expert

### B0. Phase B 稳定性保障（防崩溃）

Phase B 引入新数据时面临三重分布跳变（数据分布、路由监督、reward 分布）。
以下机制确保过渡平稳：

1. **Critic 重置 + Warmup**
   - Resume 时重置 Critic 末层（orthogonal init, gain=0.01）
   - 前 `critic_warmup_iters=3000` 步只训 Critic（surrogate loss 置零），让 value function 先适应新数据
   - Value loss 硬限 `max_value_loss=100.0`，防止 inf 导致全局崩溃

2. **旧数据锚定（Data Mixing）**
   - 旧数据（第一个 motion_dir）按 `phase_b_old_data_ratio=0.8` 初始采样比例
   - 线性退火到 `phase_b_old_data_ratio_min=0.3`（经 `phase_b_ratio_anneal_iters=30000` 步）
   - 旧数据上 Router 行为正确，稳住 Critic 和 Expert 的基础能力

3. **伪标签路由监督**
   - Phase A 的 CSV 标注作为伪标签（`phase_b_pseudo_level_csv`）
   - 旧文件保留 level 标签，新文件 level=0
   - Adaptive 模式下对有标签样本施加弱 CE loss（`adaptive_ce_coef=0.05`），防止 Router 漂移
   - 无标签样本只用 load-balance loss

### B1. 先不新增，测"能不能解释"

对未标注数据 D_u：
- 用当前 {Enc, Router, Experts} 正常训练
- Router 切换到 adaptive 模式（不受课程 mask 约束）
- 记录每个 env 的困难度指标

### B2. 新增 Expert 触发条件

per-env 困难度指标（EMA 滑动统计，alpha=0.995）：

| 指标 | 计算方式 | 阈值 |
|------|---------|------|
| tracking_error | 来自 env infos 或 -reward | `err > 0.15` |
| routing_entropy | `-sum(p * log(p))` | `entropy > 0.8` |
| top1_top2_gap | `p_top1 - p_top2` | 辅助监控 |

触发条件：`error > tau_err AND entropy > tau_H` 在 30% 的 env 中持续 50 个窗口。

### B3. 新 Expert 怎么增

1. **新增 Expert MLP**：与现有 expert 相同结构 `[512, 256]`
2. **扩展 Gate 输出层**：最后一个 Linear 层增加一个输出维度
3. **训练策略**：
   - 新 expert：`learning_rate * 3.0`（高学习率）
   - 旧 experts：`learning_rate * 0.1`（低学习率，防遗忘）
   - Router + Encoder + Critic：`learning_rate`（正常学习率）

### B4. 冷启动约束（防止新 expert 抢戏）

新增后前 `warmup_steps=5000` 步：
- 限制新 expert 最大权重 `p_new <= 0.3`
- 超出部分重新归一化分配给其他候选 expert
- 这样新 expert 在"它该负责的区域"逐渐成长，不扰乱全局

## Level 信息传递

**观测空间不变**。Level 标签通过 side channel 传递：

1. `MotionCommand.get_env_difficulty_levels()` 返回 per-env level（仅内部调用）
2. `MotionOnPolicyRunner` 在 rollout 收集时获取 level 并传给 `MoEPPO.act()`
3. `MoEPPO` 在平行 buffer 中存储 level，update 时同步 mini-batch 传给 policy

推理时不需要 level 信息，policy 自动走 adaptive routing。

## 最终部署

部署 policy = `{Enc, Router, Experts}`，推理步骤：

1. 输入 `o_t` 和未来动作窗口 `g_{t:t+H}`
2. `processed_obs, z_t = FutureCommandProcessor(g).forward_with_latent(cmd)`
3. `logits = Router(z_t)`（128-dim latent）
4. 每 M 步更新候选：`K = topk(logits, k=3)`
5. 每步 soft：`p = softmax(logits[K] / T)`
6. `a = sum(p_i * E_K[i](processed_obs))`（无 Encoder，直接用 processed_obs）
7. 输出 joint targets

## 配置参数

### policy.moe_config（通用）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_experts` | 12 | Expert 总数（10 levels + 2 spare） |
| `expert_hidden_dims` | [512, 256] | 每个 expert MLP 的隐藏层 |
| `gate_hidden_dims` | [256] | Gating Network 隐藏层 |
| `top_k` | 3 | soft top-k 候选数 |
| `routing_loss_coef` | 0.1 | CE 路由监督 loss 系数 |
| `load_balance_loss_coef` | 0.01 | Load-balance loss 系数 |
| `supervised_hard_ratio` | 0.0 | 全局硬路由概率（已废弃，设为 0） |
| `new_level_hard_ratio` | 0.8 | 新 level 样本硬路由概率（确保新 expert 学到对应特征） |
| `expert_warm_start` | true | 升级时复制上一 expert 参数（warm start 加速训练） |

### policy.moe_config（Soft Top-K 路由）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `candidate_update_interval` | 10 | 候选集更新间隔（M 步） |
| `temperature` | 0.5 | softmax 温度（T<1 更集中） |
| `logits_ema_alpha` | 0.8 | logits EMA 平滑系数（仅 rollout） |
| `use_zt_for_gate` | true | Router 是否用 z_t 做输入 |

### policy.moe_config（Phase B 动态新增）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `phase_b_enabled` | false | 是否启用 Phase B |
| `adaptive_ce_coef` | 0.05 | adaptive 模式下对有标签数据的 CE loss 系数 |
| `max_value_loss` | 100.0 | value loss 上限（防止 inf 崩溃） |
| `critic_warmup_iters` | 3000 | Phase B 开始后只训 critic 的迭代数 |
| `new_expert_weight_cap` | 0.3 | 冷启动最大权重 |
| `new_expert_warmup_steps` | 5000 | 冷启动持续步数 |
| `max_total_experts` | 16 | 最多可扩展到的 expert 数 |
| `new_expert_lr_scale` | 3.0 | 新 expert 学习率倍数 |
| `old_expert_lr_scale` | 0.1 | 旧 experts 学习率倍数 |
| `difficulty_err_threshold` | 0.15 | tracking error EMA 阈值 |
| `difficulty_entropy_threshold` | 0.8 | routing entropy EMA 阈值 |
| `difficulty_sustained_windows` | 50 | 连续困难窗口数 |
| `difficulty_trigger_ratio` | 0.3 | 触发所需 env 困难比例 |
| `difficulty_check_interval` | 500 | 每 N iter 检查一次 |

### commands.motion Phase B 数据混合

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `phase_b_old_data_ratio` | 0.8 | 初始旧数据采样比例（0=不启用） |
| `phase_b_old_data_ratio_min` | 0.3 | 退火后的最低比例 |
| `phase_b_ratio_anneal_iters` | 30000 | 线性退火迭代数 |
| `phase_b_pseudo_level_csv` | null | Phase A 的课程 CSV 路径（用于伪标签） |

### commands.motion 课程升级与采样

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `curriculum_auto_upgrade_per_level` | null | Per-level 硬超时列表（index 0=lv1） |
| `curriculum_reward_convergence_enabled` | true | 是否启用 reward 收敛提前升级 |
| `curriculum_reward_convergence_window` | 20 | reward 滑动窗口大小 |
| `curriculum_reward_convergence_threshold` | 0.02 | 相对改善率阈值 |
| `curriculum_reward_convergence_min_iters` | 3000 | 收敛升级最低训练迭代数 |
| `curriculum_level_difficulty_weights` | null | 按 level 的采样权重倍数列表 |

### commands.motion MoE 专用

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `moe_supervised_until_level` | 10 | 达到此 level 后切换到 adaptive |
| `moe_extra_unlabeled_dirs` | null | Phase B 引入的无标注数据目录列表 |

## 训练命令

### Phase A：课程约束训练

```bash
conda activate bjc@beyondmimic
export CUDA_VISIBLE_DEVICES=1,2
export DISPLAY=:2
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --g1_training_cfg source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/cfg/g1_training_moe_phase_a.yaml \
  --motion_dir /ssd/humanoid/dataset/humanml_gmr/npz \
  --num_envs 8192 \
  --headless \
  --logger wandb \
  --log_project_name bmtest \
  --run_name moe_phase_a_soft_top3_conlv7_211csv \
  --no_wandb_model_upload \
  --resume True \
  --load_run 2026-02-21_00-34-34_moe_phase_a_soft_top3_conlv6_211csv \
  --checkpoint model_97000.pt
```

### Phase B：从 Phase A checkpoint 继续

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --g1_training_cfg source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/cfg/g1_training_moe_phase_b.yaml \
  --motion_dir /ssd/humanoid/dataset/humanml_gmr/npz  /ssd/humanoid/dataset/lafan_npz\
  --num_envs 8192 \
  --headless \
  --logger wandb \
  --log_project_name bmtest \
  --run_name moe_phase_b_soft_top3_gmr_lafan\
  --no_wandb_model_upload \
  --resume True \
  --load_run 2026-03-07_12-21-49_moe_phase_a_soft_top3_conlv7_211csv \
  --checkpoint model_187000.pt
```
### Phase B：从 Phase B checkpoint 继续
```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --g1_training_cfg source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/cfg/g1_training_moe_phase_b.yaml \
  --motion_dir /ssd/humanoid/dataset/humanml_gmr/npz /ssd/humanoid/dataset/lafan_npz \
  --num_envs 8192 \
  --headless \
  --logger wandb \
  --log_project_name bmtest \
  --run_name moe_phase_b_soft_top3_gmr_lafan_resume \
  --no_wandb_model_upload \
  --resume True \
  --load_run 2026-04-04_00-03-57_moe_phase_b_soft_top3_gmr_lafan \
  --checkpoint model_286999.pt
```

`train.py` 不需要任何修改。`MotionOnPolicyRunner` 会自动检测 policy 类型：
- 当 `class_name: "EncoderActorCritic"` 时：走标准 MLP 路径（与现有行为完全一致）
- 当 `class_name: "MoEActorCritic"` 时：自动升级为 MoE 训练模式

## Checkpoint 兼容性

- MoE 模型结构与 MLP 不同，**不能直接 resume MLP 旧 checkpoint**
- Phase B 可以 resume Phase A 的 checkpoint（权重兼容）
- Phase B 新增 expert 后 checkpoint 会包含更多 expert 权重
- MoE checkpoint 包含 `routing_mode` 和 `max_active_experts` 信息

## 关键文件

| 文件 | 说明 |
|------|------|
| `encoder_actor_critic.py` | `MoEActorCritic` 类（soft top-k routing, z_t gate, Phase B add_expert） |
| `moe_ppo.py` | `MoEPPO` 算法（PPO + level buffer + aux loss + difficulty tracking） |
| `my_on_policy_runner.py` | `MotionOnPolicyRunner`（Phase B 检查与 expert 新增触发） |
| `g1_training_moe_phase_a.yaml` | Phase A 训练配置（课程约束 soft top-k） |
| `g1_training_moe_phase_b.yaml` | Phase B 训练配置（动态新增 expert） |
| `g1_training_moe.yaml` | 旧配置（保留作为参考/回退） |
| `commands.py` | `get_env_difficulty_levels()` 方法 |
| `curriculum_manager.py` | 课程管理器 |

## 性能特性

- **计算优化**：只 forward 候选的 k=2~3 个 expert（非全部 12 个），通过 `torch.unique` + `torch.searchsorted` + `gather` 批量操作
- **显存优化**：不存储全 expert 输出，只存候选 expert 的 `(B, k, A)`
- **路由轻量**：Gate 输入仅 128-dim（z_t），远小于 encoded_obs；低频更新进一步降低开销
- **数据分离**：`preload_device: "cuda:1"` 逻辑不变，motion 数据仍在独立 GPU

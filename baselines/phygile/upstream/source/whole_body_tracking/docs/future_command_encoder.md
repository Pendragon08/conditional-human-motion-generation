# 卷积式命令编码器使用说明

本说明整理了 tracking 任务中新加的 “未来命令卷积编码” 流程，方便在不同配置中启用或扩展。

## 1. 观测拆分方式

`mdp.generated_commands` 现在输出三段连续帧：

| 段落 | 来源 | 作用 |
| --- | --- | --- |
| `gt` 当前帧 | `joint_pos + joint_vel + root_pos + root_quat` | 始终裸露给 policy，提供即时目标 |
| `future_steps` 帧 | `commands.motion.future_steps`（短 horizon） | 直接拼在观测里，保持精细跟踪；每帧包含 `joint_pos`, `joint_vel`, `root_pos`, `root_quat` |
| `encoder_future_steps` 帧 | `commands.motion.encoder_future_steps`（长 horizon，可选） | 仅在启用卷积 encoder 时采样，同样包含关节+root 信息，用于提取长期趋势 |

最终拼接顺序为 `[gt] + [future_steps * per_frame_dim] + [encoder_future_steps * per_frame_dim]`，其中 `per_frame_dim = 2 * joint_dof + 7`（关节位置/速度 + root xyz + root 四元数）。如果 `encoder_future_steps` 省略或为 0，则只保留前两段；若大于 0，则卷积 encoder 将最后一段看成时序序列（形状 `(B, per_frame_dim, T)`）做一维卷积，输出压缩的 `z_t`。

## 2. MotionCommand 配置

```yaml
commands:
  motion:
    future_steps: 2            # 暴露给 policy 的短期帧数，可为 0
    encoder_future_steps: 10   # 卷积 encoder 实际消费的帧数，可为 0
    encoder_future_stride: 10  # 可选，下采样时的步长；等于 10 表示覆盖 ~100 帧的窗口但仅采样 10 帧
```

- 两个值都为 0 时，退化为原先的“单帧命令”。
- `encoder_future_steps` 不影响 policy 输入维度，若不启用卷积 encoder，可以保持 0。

## 3. Policy 侧配置

`policy.future_command_encoder` 控制卷积结构；如果该字段缺失，则策略端保持完全原有的 MLP 行为：

```yaml
policy:
  future_command_encoder:
    per_frame_dim: null          # 可选，默认 2 * num_actions + 7
    immediate_steps: 2           # 与 commands.motion.future_steps 保持一致
    conv_steps: 100              # 与 commands.motion.encoder_future_steps 保持一致
    conv_channels: [512, 512, 256]
    kernel_size: 3
    pooling: "avg"               # "avg" 或 "max"
    dropout: 0.0
    latent_dim: 128              # 卷积 + 池化后映射为固定维度 z_t
    share_across_actor_critic: true
    apply_to_actor: true
    apply_to_critic: true
```

关键说明：

- `immediate_steps` + 1（当前帧）组成原样拼接段；`conv_steps` 表示进入时序卷积的帧数（可省略，此时默认使用同一块配置中的 `future_steps` 字段）。
- `share_across_actor_critic=true` 时，actor/critic 复用同一卷积编码器；否则分别实例化。
- `apply_to_actor/critic` 可独立关闭任一路，便于做 ablation。
- `per_frame_dim` 若未显式给出，会根据动作维度自动推算（`2 * num_actions + 7`，其中额外的 7 表示 root xyz + 四元数）。如命令格式发生变化，请显式填写。
- `latent_dim` 控制最终 `z_t` 的长度（默认 128）。如果希望直接使用卷积输出通道，可将该字段设为 `null`。
- `encoder_future_stride` 允许在命令端稀疏采样（默认 1）。例如 `encoder_future_steps=10、encoder_future_stride=10` 表示覆盖 100 帧窗口但只送入 10 帧，显著降低显存与算力开销。此时 policy 侧只需将 `conv_steps` 设为 10。

## 4. 内部实现概览

1. `MotionCommand.command` 会根据 `future_steps` 与 `encoder_future_steps` 采样对应帧，确保多文件模式下仍能正确对齐。
2. `EncoderActorCritic` 新增 `_FutureCommandProcessor`，在进入 MLP encoder 之前：
   - 取出前缀 `[current + immediate]`，直接拼入观测。
   - 将后续 `conv_steps` 帧 reshape 成 `(B, per_frame_dim, conv_steps)`，沿时间轴做 `Conv1d + 激活 (+ Dropout)`，然后 `Adaptive{Avg/Max}Pool1d(1)`，并用 `Linear` 压缩为指定的 `latent_dim`（默认 128）即 `z_t`。
   - 将 `[current + immediate] + z_t` 与剩余 proprio/privileged 观测重新拼接，再进入共享/独立 MLP encoder。
3. 若配置与命令帧数不匹配，会在运行时抛出错误，避免 silent bug。若想完全关闭该功能，只需移除 `policy.future_command_encoder`，并/或将 `commands.motion.encoder_future_steps` 设为 0（默认值）。

## 5. 示例：`g1_training_future_conv_larger_freeze.yaml`

```yaml
policy:
  future_command_encoder:
    per_frame_dim: 65       # 2 * joint_pos_dim (29) + root xyz/quaternion
    immediate_steps: 2      # 与 commands.motion.future_steps 保持一致
    conv_steps: 5           # 与 commands.motion.encoder_future_steps 保持一致
    conv_channels: [256, 128]
    kernel_size: 3
    pooling: "avg"
    dropout: 0.0
    latent_dim: 128
    share_across_actor_critic: true
    apply_to_actor: true
    apply_to_critic: true
commands:
  motion:
    future_steps: 2
    encoder_future_steps: 5
    encoder_future_stride: 20   # 覆盖 5*20=100 帧窗口，但只采样 5 帧送入卷积
```

## 6. 调试与常见问题

- **Dimension mismatch**：确保 `immediate_steps == future_steps` 且 `conv_steps == encoder_future_steps`。否则 `_FutureCommandProcessor` 会直接报错。
- **旧模型兼容性**：由于 observation layout 发生变化，旧策略权重需要重新训练。
- **性能调参**：
  - 减小 `conv_steps` 或通道数可降低显存占用。
  - 将 `apply_to_critic=false` 可以只在 actor 侧使用卷积 encoder，以测试其对评估网络的影响。
  - `pooling` 选择 `max` 可更强调峰值动作，`avg` 则平滑趋势。

如需进一步扩展（例如改用 Transformer encoder），可复用 `_FutureCommandProcessor` 的切分逻辑，只需在 `EncoderActorCritic` 中替换对应处理器即可。

